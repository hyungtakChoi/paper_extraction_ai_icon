import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import load_from_disk
from transformers import default_data_collator
from transformers import get_linear_schedule_with_warmup # 학습률 스케줄러
import numpy as np
import mmap
import json
from tqdm import tqdm
import time

import brain.work.arch.transformer as transformer
import brain.work.arch.util as archutil

class PE(nn.Module):
    """Positional embedding (learnable)."""

    def __init__(self, ctx_window, d_emb):
        super().__init__()
        self.position_embeddings = nn.Embedding(ctx_window, d_emb)

    def forward(self, x): # x: (B, C, E)
        B, C, _ = x.size()
        position_ids = torch.arange(C, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(B, C)  # (B, C)
        pos_emb = self.position_embeddings(position_ids)  # (B, C, E)
        return pos_emb

class BertInputEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_emb, max_sequence_length, type_vocab_size, dropout_rate):
        # type_vocab_size = 2
        super().__init__()
        self.token_embeddings = transformer.ELUT(vocab_size, d_emb)
        self.position_embeddings = PE(max_sequence_length, d_emb)
        self.segment_embeddings = nn.Embedding(type_vocab_size, d_emb)
        self.layer_norm = nn.LayerNorm(d_emb, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids, token_type_ids=None):
        # input_ids: (B, C)
        # token_type_ids: (B, C)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        token_emb = self.token_embeddings(input_ids)         # (B, C, E)
        pos_emb = self.position_embeddings(token_emb)        # (B, C, E)
        seg_emb = self.segment_embeddings(token_type_ids)    # (B, C, E)

        embeddings = token_emb + pos_emb + seg_emb
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertEncoder(nn.Module):
    def __init__(self, num_layers, d_emb, d_q, d_k, d_ff, n_heads, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList([
            transformer.EncoderLayer(d_emb, d_q, d_k, d_ff, n_heads, dropout_rate)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class BertModel(nn.Module):
    def __init__(self, bert_config, use_pooler=False, use_classifier=False, num_classes=None):
        super().__init__()

        self.ctx_window_enc = bert_config.ctx_window_enc

        self.embeddings = BertInputEmbeddings(
            bert_config.vocab_size, bert_config.d_emb, bert_config.ctx_window_enc, 2, bert_config.dropout_rate_enc
        )
        self.encoder = BertEncoder(
            bert_config.n_layers_enc, bert_config.d_emb, bert_config.d_q, bert_config.d_k, bert_config.d_ff, bert_config.n_heads_enc, bert_config.dropout_rate_enc
        )
        # (선택) Pooler, Classifier 등 추가
        self.use_pooler = use_pooler
        self.use_classifier = use_classifier
        if self.use_pooler:
            self.pooler = nn.Sequential(
                nn.Linear(bert_config.d_emb, bert_config.d_emb),
                nn.Tanh()
            )
        else:
            self.pooler = None

        if self.use_classifier:
            assert num_classes is not None, "num_classes must be specified if use_classifier is True."
            self.classifier = nn.Linear(bert_config.d_emb, num_classes)
        else:
            self.classifier = None

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        input_ids = archutil.crop_data_to_ctx_window(input_ids, self.ctx_window_enc)

        x = self.embeddings(input_ids, token_type_ids)
        # print("after embedding nan:", torch.isnan(x).any().item())
        # if attention_mask is None:
        #     attention_mask = create_pad_mask(input_ids)

        # attention_mask = attention_mask.unsqueeze(1).bool()
        enc_padding_mask = (attention_mask == 0).unsqueeze(-1)
        # attention_mask = create_pad_mask(input_ids)
        x = self.encoder(x, enc_padding_mask)
        # print("after encoder nan:", torch.isnan(x).any().item())
        outputs = {"sequence_output": x}

        cls_token_output = x[:, 0]
        
        if self.use_pooler:
            pooled_output = self.pooler(cls_token_output)
            outputs["pooled_output"] = pooled_output

        # Classifier를 사용할 경우, 로직을 수행합니다.
        if self.use_classifier:
            # Pooler가 사용되었다면 pooled_output을, 아니라면 [CLS] 토큰의 원본 출력을 사용합니다.
            if self.use_pooler:
                classifier_input = outputs["pooled_output"]
            else:
                classifier_input = cls_token_output
                
            logits = self.classifier(classifier_input)
            outputs["logits"] = logits

        return outputs

class BertMLMHead(nn.Module):
    def __init__(self, d_emb, vocab_size):
        super().__init__()
        self.dense = nn.Linear(d_emb, d_emb)
        self.layer_norm = nn.LayerNorm(d_emb, eps=1e-12)
        self.decoder = nn.Linear(d_emb, vocab_size, bias=False)
        # # bias for decoder
        # self.bias = nn.Parameter(torch.zeros(vocab_size))
        # self.decoder.bias = self.bias

    def forward(self, sequence_output):
        # sequence_output: (batch, seq_len, d_emb)
        x = self.dense(sequence_output)
        # x = torch.relu(x)
        x = F.gelu(x)
        x = self.layer_norm(x)
        logits = self.decoder(x)  # (batch, seq_len, vocab_size)
        return logits

class BertNSPHead(nn.Module):
    def __init__(self, d_emb):
        super().__init__()
        self.classifier = nn.Linear(d_emb, 2)  # 2-class (is_next, not_next)

    def forward(self, pooled_output):
        # pooled_output: (batch, d_emb)  # [CLS] 임베딩
        logits = self.classifier(pooled_output)  # (batch, 2)
        return logits
