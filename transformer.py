"""Transformer-base.

Notes
-----
1. (Notation: `B`, `C`) In both the code and the documentation, `B` and
`C` are actually variables, not constants like `tr_batch_size`
`ctx_window_enc`, and `ctx_window_dec`. `B` is either 1 (during
generation) or `batch_size` (during training). `C` may vary from 1 to
some_ctx_window as well.
"""

import math

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

import transformer_tokenizer as tfmtok
import arch_util as archutil

# TODO (before Transformer):
#   1. Probably construct a vocabulary union (NMT problem).
#   2. Tokenization
#   3. Data preprocessing (including <SOS>, etc.)
#     - Output data (shifted right); Fig. 1 of the 2017 paper
#   4. Padding/truncation
#
# TODO (in Transformer)
#   1. Fix masking.
#     a. Mask in the rectangular attn pattern setup
#     b. Shape of mask changes all the time; a single global mask does
#        not work. [GC: I think we only need to fix the `generate` part
#        in this respect.]
#     (* Mask in HF Transformers Tokenizer seems to be only for MLM
#        problems.)
#   2. Double-check if there's something to fix w.r.t. rectangular
#      attn pattern.

class ELUT(nn.Module):
    """Embedding look-up table."""

    def __init__(self, vocab_size, d_emb):
        super().__init__()
        # lookup table whose learnable weights are a tensor of size
        # (vocab_size, E)
        self.lut = nn.Embedding(vocab_size, d_emb)
        self.d_emb = d_emb

    def forward(self, x):
        """Create embeddings.

        Parameters
        -----------
        x : torch.Tensor
            Tokens of size (B, C).

        Returns
        -------
        torch.Tensor
            Embeddings of size (B, C, E).
        """
        return self.lut(x) * math.sqrt(self.d_emb) # (B, C, E)

class PE(nn.Module):
    """Positional encoding."""

    def __init__(self, ctx_window, d_emb, dropout_rate):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        # Compute PE at once using tensor, in log space.
        pe = torch.zeros(ctx_window, d_emb) # (ctx_window, E)
        pos = torch.arange(0, ctx_window).unsqueeze(1) # (ctx_window, 1)
        div = torch.exp( # (E/2)
            torch.arange(0, d_emb, 2) * -(math.log(10000.) / d_emb)
        )
        pe[:, 0::2] = torch.sin(pos * div) # broadcasting; (ctx_window, E/2)
        pe[:, 1::2] = torch.cos(pos * div) # broadcasting; (ctx_window, E/2)
        pe = pe.unsqueeze(0) # for batch; (1, ctx_window, E)
        self.register_buffer("pe", pe)

    def forward(self, x): # x: (B, length, E)
        torch._assert(
            x.shape[1] <= self.pe.shape[1],
            ("Length of input sequence must not be greater than PE "
            "length (context window)")
        )
        pe = self.pe[:, :x.shape[1], :].requires_grad_(False) # (1, length, E)
        x = x + pe # (B, length, E); B-broadcasting
        return self.dropout(x)

def gen_mask(B, C):
    """Return a mask tensor, whose values are Boolean.

    True means that the corresponding value should be masked, and False
    means we keep the original value. The returned mask tensor has the same
    size as attention pattern."""

    triu = torch.ones(B, C, C).triu()
    return triu == 0 # (B, C, C)

def gen_numb_mask(B, C):
    """Return a numb mask, where all the values are False.

    Use this mask as a placeholder when there is nothing to mask."""

    return torch.ones(B, C, C) == 0 # (B, C, C)

class SHA(nn.Module):
    """Single-head attention.

    1. This class becomes either self- or cross-attention, depending on
    the input data (x_q, x_k, x_v) to `forward`; it is self-attention
    if x_q, x_k, x_v are all from the same source, it is
    cross-attention if x_q is from one source and x_k and x_v are from
    another.

    2. This class becomes either an encoder SHA or a decoder SHA,
    depending on the `mask` to `forward`; it is an encoder SHA if the
    given `mask` is a numb mask (i.e., a tensor whose elements are all
    False), it is a decoder SHA if an actual mask (i.e., a tensor whose
    upper-triangular parts including the diagonals are all True and the
    lower-triangular parts excluding the diagonals are all False) is
    given."""

    def __init__(self, d_emb, d_q, d_k):
        super().__init__()
        #  nn.Linear is expressed rowwise, but it is defined columnwise.
        self.W_Q = nn.Linear(d_emb, d_q, bias=False)
        self.W_K = nn.Linear(d_emb, d_k, bias=False)
        self.W_V = nn.Linear(d_emb, d_emb, bias=False)
        self.d_q = d_q

    def forward(self, x_q, x_k, x_v, mask):
        """Compute single-head attention.

        Parameters
        ----------
        x_q : torch.Tensor
            Input data (embeddings) of size (B, C_tgt, E) for computing queries.
        x_k : torch.Tensor
            Input data (embeddings) of size (B, C_src, E) for computing keys.
        x_v : torch.Tensor
            Input data (embeddings) of size (B, C_src, E) for computing values.
        mask : torch.Tensor

        Returns
        -------
        torch.Tensor
            Delta-embeddings of size (B, C_tgt, E).

        Notes
        -----
        1. (source & target; `C_src` & `C_tgt`) When it is
        self-attention, the source and target languages are the same,
        and so `C_src` = `C_tgt`. When it is cross-attention, `C_src`
        and `C_tgt` are different in general.
        """
        # FIXME FIXME FIXME: commented out the assertion below
        # temporarily for playing around with torchinfo (while
        # executing torchinfo.summary, the assertion failed due to NaN
        # in `x_v`).
        '''
        torch._assert(
            torch.equal(x_k, x_v),
            "The same input sequence must be used for keys and values"
        )
        '''

        Q = self.W_Q(x_q) # (B, C_tgt, d_q)
        K = self.W_K(x_k) # (B, C_src, d_k)
        V = self.W_V(x_v) # (B, C_src, E) # NOTE: naive computation w/o W_O; E = d_emb

        product = torch.matmul(K, Q.transpose(-2, -1)) / math.sqrt(self.d_q) # (B, C_src, C_tgrt)
        product = product.masked_fill(mask, float('-inf')) # (B, C_src, C_tgt)
        pattern = F.softmax(product, dim=-2) # (B, C_src, C_tgt)
        attn = torch.matmul(V.transpose(-2, -1), pattern) # (B, E, C_tgt)
        attn = attn.transpose(-2, -1) # (B, C_tgt, E)
        return attn # delta-E

class MHA(nn.Module):
    """Multi-head attention."""

    def __init__(self, d_emb, d_q, d_k, n_heads):
        super().__init__()
        self.heads = nn.ModuleList([SHA(d_emb, d_q, d_k) for _ in range(n_heads)])
    
    def forward(self, x_q, x_k, x_v, mask):
        """Compute multi-head attention.

        Parameters
        ----------
        x_q : torch.Tensor
            Input data (embeddings) of size (B, C_tgt, E) for computing queries.
        x_k : torch.Tensor
            Input data (embeddings) of size (B, C_src, E) for computing keys.
        x_v : torch.Tensor
            Input data (embeddings) of size (B, C_src, E) for computing values.
        mask : torch.Tensor

        Returns
        -------
        torch.Tensor
            Delta-embeddings of size (B, C_tgt, E).

        Notes
        -----
        1. (Shape of attention pattern) Attention pattern has a
        rectangular shape in general; it does not have to be a square
        matrix. An attention pattern in self-attention is usually
        square. An attention pattern in cross-attention is a
        rectangular matrix in general. That is the case regardless of
        whether it is at the training or generation phase. During
        training, input and output sequences (from two different
        languages) may have different lengths. During generation (or
        inference), the length of the input sequence to the decoder
        (i.e., sequence that is being generated) grows from 1 to
        some_max_size in an auto-regressive manner, while the length of
        the sequence from the encoder (a.k.a. memory) stays the same.

        2. (Size of attention pattern) The size of attention pattern
        changes constantly across batches in mini-batch training. That
        is because input sequences in a batch are all set to have the
        length of the longest sequence of the batch (or context window)
        via padding/truncation, and the longest length may differ
        across batches. The size of attention pattern changes across
        generation steps in the decoder as well. That is because the
        size of attention pattern is determined by the length of the
        input sequence, and the length of the input sequence (i.e.,
        from the target language) to the decoder grows from 1 (usually
        <SOS>) until <EOS> is sampled or the size becomes some_max_len.
        """
        xs = torch.stack( # (n_heads, B, C_tgt, E)
            [h(x_q, x_k, x_v, mask) for h in self.heads], dim=0
        )
        x = torch.sum(xs, dim=0, keepdim=False) # (B, C_tgt, E)
        return x

class FF(nn.Module):
    """Fully connected feedforward network, or MLP."""

    def __init__(self, d_emb, d_ff):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_emb, d_ff, bias=True),
            nn.ReLU(), 
            nn.Linear(d_ff, d_emb, bias=True)
        )

    def forward(self, x):
        """Compute fully connected feedforward network.

        Parameters
        ----------
        x : torch.Tensor
            Embeddings of size (B, C, E).

        Returns
        -------
        torch.Tensor
            Embeddings of size (B, C, E).
        """
        return self.ff(x) # (B, C, E)

class SublayerConnection(nn.Module):
    """Residual connection followed by a layer normalization."""

    def __init__(self, d_emb, dropout_rate):
        super().__init__()
        self.norm = nn.LayerNorm([d_emb]) # same as `nn.LayerNorm(d_emb)`
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, sublayer):
        """

        Parameters
        ----------
        x : torch.Tensor
            Embeddings of size (B, C, E).
        sublayer : callable
            Function that implements the sublayer.
            Expected signature: (embeddings: torch.Tensor) -> torch.Tensor

        Returns
        -------
        torch.Tensor
            Embeddings of size (B, C, E).
        """
        sub_out = sublayer(x) # (B, C, E)
        sub_out = self.dropout(sub_out) # (B, C, E)
        res = x + sub_out # (B, C, E)
        return self.norm(res)

class EncoderLayer(nn.Module):

    def __init__(self, d_emb, d_q, d_k, d_ff, n_heads, dropout_rate):
        super().__init__()
        self.mhsa = MHA(d_emb, d_q, d_k, n_heads)
        self.ff = FF(d_emb, d_ff)
        self.slc = nn.ModuleList([SublayerConnection(d_emb, dropout_rate) for _ in range(2)])

    def forward(self, x, mask):
        """Compute an encoder layer.

        Parameters
        ----------
        x : torch.Tensor
            Embeddings of size (B, C_src, E).
        mask : torch.Tensor
            Numb mask for self-attention inside the encoder layer
            (i.e., no masking).

        Returns
        -------
        torch.Tensor
            Embeddings of size (B, C_src, E).
        """
        x = self.slc[0](x, lambda x: self.mhsa(x, x, x, mask)) # (B, C_src, E)
        x = self.slc[1](x, self.ff) # (B, C_src, E)
        return x

class DecoderLayer(nn.Module):

    def __init__(self, d_emb, d_q, d_k, d_ff, n_heads_sa, n_heads_ca, dropout_rate):
        super().__init__()
        # Multi-head self-attention (inside this decoder layer)
        self.mhsa = MHA(d_emb, d_q, d_k, n_heads_sa)
        # Multi-head cross-attention (between the whole encoder and
        # this decoder layer)
        self.mhca = MHA(d_emb, d_q, d_k, n_heads_ca)
        self.ff = FF(d_emb, d_ff)
        self.slc = nn.ModuleList(SublayerConnection(d_emb, dropout_rate) for _ in range(3))

    def forward(self, x, x_enc, mask_sa, mask_ca):
        """Compute a decoder layer.

        Parameters
        ----------
        x : torch.Tensor
            Normal input data (embeddings) inside the decoder layer, of
            size (B, C_tgt, E).
        x_enc : torch.Tensor
            Input data (embeddings) from the encoder, which is often
            called the 'memory', of size (B, C_src, E).
        mask_sa : torch.Tensor
            Mask for self-sttention inside the decoder layer (i.e.,
            actual masking is applied in order to preserve the
            auto-regressive property).
        mask_ca : torch.Tensor
            Numb mask for cross-attention between the encoder and each
            decoder layer (i.e., no masking).

        Returns
        -------
        torch.Tensor
            Tensor of size (B, C_tgt, E).
        """
        # Masked multi-head self-attention
        x = self.slc[0](x, lambda x: self.mhsa(x, x, x, mask_sa)) # (B, C_tgt, E)
        # Multi-head cross-attention without masking
        x = self.slc[1](x, lambda x: self.mhca(x, x_enc, x_enc, mask_ca)) # (B, C_tgt, E)
        x = self.slc[2](x, self.ff) # (B, C_tgt, E)
        return x

class Encoder(nn.Module):
    
    def __init__(self, d_emb, d_q, d_k, d_ff, n_heads, n_layers, dropout_rate):
        super().__init__()
        self.enc = nn.ModuleList(
            [EncoderLayer(d_emb, d_q, d_k, d_ff, n_heads, dropout_rate)
                for _ in range(n_layers)]
        )

    def forward(self, x, mask):
        """Compute the encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input data (embeddings) to the encoder, of size
            (B, C_src, E).
        mask : torch.Tensor
            Numb mask for self-attention inside the encoder (i.e., no
            masking).

        Returns
        -------
        torch.Tensor
            Embeddings of size (B, C_src, E).
        """
        for layer in self.enc:
            x = layer(x, mask) # (B, C_src, E)
        return x

class Decoder(nn.Module):

    def __init__(self, d_emb, d_q, d_k, d_ff, n_heads_sa, n_heads_ca, n_layers, dropout_rate):
        super().__init__()
        self.dec = nn.ModuleList(
            [DecoderLayer(d_emb, d_q, d_k, d_ff, n_heads_sa, n_heads_ca, dropout_rate)
                for _ in range(n_layers)]
        )

    def forward(self, x, x_enc, mask_sa, mask_ca):
        """Compute the decoder.

        Parameters
        ----------
        x : torch.Tensor
            Normal input data (embeddings) of size (B, C_tgt, E) to the
            decoder.
        x_enc : torch.Tensor
            Input data (embeddings) of size (B, C_src, E) from the
            encoder, which is often called the 'memory'.
        mask_sa : torch.Tensor
            Mask of size (B, C, C) for self-attention inside the
            decoder (i.e., actual masking is applied in order to
            preserve the auto-regressive property).
        mask_ca : torch.Tensor
            Numb mask of size (B, C, C) for cross-attention between the
            encoder and each decoder layer in the decoder (i.e., no
            masking).

        Returns
        -------
        torch.Tensor
            Embeddings of size (B, C_tgt, E).
        """
        for layer in self.dec:
            x = layer(x, x_enc, mask_sa, mask_ca) # (B, C_tgt, E)
        return x

def gen_token_greedy(probs):
    """Generate a new token by greedy decoding.

    Parameters
    ----------
    probs : torch.Tensor
        Probabilities of size (B, 1, V).

    Returns
    -------
    torch.Tensor of dtype torch.int32
        Sampled token (i.e., index) of size (B, 1).
    """
    token = torch.max(probs, dim=-1, keepdim=False).values.int() # (B, 1)
    return token

def gen_token_multinomial(probs):
    """Generate a new token by multinomial sampling.

    Parameters
    ----------
    probs : torch.Tensor
        Probabilities of size (B, 1, V).

    Returns
    -------
    torch.Tensor
        Sampled token of size (B, 1).
    """
    probs = probs.squeeze(dim=-2) # (B, V)
    token = torch.multinomial(probs, 1) # (B, 1)
    return token

def is_eos(x):
    return tfmtok.is_eos(x)

class Transformer(nn.Module):
    """Transformer-base (2017).

    Notes
    -----
    Assumption A (tokenization). Tokenization is done before a
    `Transformer` instance.

    Assumption B (padding/truncation). Padding/truncation is done
    before a `Transformer` instance.

    1. (Shared ELUT) Transformer-base shares the same weights (and
    vocabs.) among the following three parts: (1) source language
    sequence embedding, (2) target language sequence embedding, and (3)
    the final linear layer for computing logits [1]_. The sharing is a
    common practice for similar languages like English and French. What
    enables the sharing includes subword tokenization algorithms such
    as WordPiece and BPE[2]_, and vocabulary union.

    2. (Number of executions: encoder vs. decoder) During generation
    (i.e., inference), the encoder executes only once, and the decoder
    executes #some_max_size times with growing inputs (from the target
    language) to the decoder in an auto-regressive manner.

    3. (Transformer general fact: padding vs. truncation) Sequences in
    a batch often have different lengths. In order to process them in
    fixed-sized tensors throughout, padding/truncation strategies are
    used [3]_[4]_[5]_.

    References
    ----------
    .. [1] Vaswani, Ashish, et al. "Attention is all you need."
       Advances in neural information processing systems 30 (2017).
    .. [2] Xia, Yingce, et al. "Tied transformers: Neural machine
       translation with shared encoder and decoder." Proceedings of the
       AAAI conference on artificial intelligence. Vol. 33. No. 01.
       2019.
    .. [3] https://huggingface.co/docs/transformers/en/pad_truncation
    .. [4] https://medium.com/@g.martino8/all-the-questions-about-transformer-model-answered-part-5-the-padding-mask-73be0941bc1e
    .. [5] https://huggingface.co/learn/llm-course/chapter2/5
    """

    def __init__(self, config):
        super().__init__()

        self.ctx_window_enc = config.ctx_window_enc
        self.ctx_window_dec = config.ctx_window_dec

        # building blocks of transformer-base
        self.elut = ELUT( # shared for both source and target languages
            config.vocab_size,
            config.d_emb
        )
        self.pe_enc = PE(
            config.ctx_window_enc,
            config.d_emb,
            config.dropout_rate_enc
        )
        self.pe_dec = PE(
            config.ctx_window_dec,
            config.d_emb,
            config.dropout_rate_dec
        )
        self.encoder = Encoder(
            config.d_emb,
            config.d_q,
            config.d_k,
            config.d_ff,
            config.n_heads_enc,
            config.n_layers_enc,
            config.dropout_rate_enc
        )
        self.decoder = Decoder(
            config.d_emb,
            config.d_q,
            config.d_k,
            config.d_ff,
            config.n_heads_dec_sa,
            config.n_heads_dec_ca,
            config.n_layers_dec,
            config.dropout_rate_dec
        )
        self.linear = self.elut.lut.weight # (V, E)

    def forward(self, x_src, x_tgt, mask, numb_mask):
        """Compute Transformer.

        The forward function returns logits, the output from
        the final linear layer before the last softmax.
 
        Parameters
        ----------
        x_src : torch.IntTensor
            Tokens of size (B, C_src). C_src is assumed to be not
            greater than the context window of the encoder.
        x_tgt : torch.IntTensor
            Tokens of size (B, C_tgt). C_tgt is assumed to be not
            greater than the context window of the decoder.
        mask : torch.Tensor
            Mask of size (B, C, C) for self-attention inside the
            decoder (i.e., actual masking is applied in order to
            preserve the auto-regressive property).
        numb_mask : torch.Tensor
            Numb mask (i.e., no masking) of size (B, C, C) either (1)
            for self-attention inside the encoder; or (2)
            cross-attention between the encoder and each decoder layer.
            
        Returns
        -------
        torch.Tensor
            Logits of size (B, C_tgt, V)

        Notes
        -----
        1. (Logits for What) It returns the logits from all input token
        positions, not just from the last position (as in
        `self.generate`). This is because the losses are computed
        across an extended mini-batch, in parallel, where (B, C_tgt) is
        the extended batch size. The extension of batch from size (B)
        to size (B, C_tgt) is aligned with the fact that we generate
        C_tgt training data from one sequence of length C_tgt [1]_.
        Also, this is the case regardless of whether we are solving a
        NMT (usually w/ encoder-decoder architecture) or NTP (usually
        w/ decoder-only architecture) problem.

        2. (Teacher forcing) During training for NMT problems, teacher
        forcing is often used. With teacher forcing, the model takes
        the true previous target tokens (as input to the decoder) rather
        than its own token predictions.

        3. (Cross Entropy Loss) The `forward` returns a logits of size
        (B, C_tgt, V), but `nn.CrossEntopyLoss` takes the input of size
        (B, V, C_tgt) in our case. Make sure to permute the dimensions
        accordingly before computing the loss [2]_[3]_.

        References
        ----------
        .. [1] Google search AI: "in a minibatch setting, torch cross entrophy loss actually computes loss over batch times sequence length"
        .. [2] Google search AI: "how to use torch cross entropy loss for transformer"
        .. [3] https://discuss.pytorch.org/t/how-to-use-crossentropyloss-on-a-transformer-model-with-variable-sequence-length-and-constant-batch-size-1/157439
        """
        x_src = archutil.crop_data_to_ctx_window(x_src, self.ctx_window_enc)

        # Constraints on the length of src & tgt sequences
        torch._assert(
            x_src.shape[1] <= self.ctx_window_enc,
            ("Length of input sequence from src must not be greater "
            "than the context window for the encoder")
        )
        torch._assert(
            x_tgt.shape[1] <= self.ctx_window_dec,
            ("Length of input sequence from tgt must not be greater "
            "than the context window for the decoder")
        )

        xs = self.elut(x_src) # (B, C_src, E)
        xs = self.pe_enc(xs) # (B, C_src, E)
        xs = self.encoder(xs, numb_mask) # (B, C_src, E)

        xt = self.elut(x_tgt) # (B, C_tgt, E)
        xt = self.pe_dec(xt) # (B, C_tgt, E)
        xt = self.decoder(xt, xs, mask, numb_mask) # (B, C_tgt, E)
        xt = torch.matmul(self.linear, xt.transpose(-2, -1)) # (V, E) @ (B, E, C_tgt) -> (B, V, C_tgt); B-broadcasting
        xt = xt.transpose(-2, -1) # (B, C_tgt, V)
        return xt

    def generate(self, x_src, x_tgt, mask, numb_mask, max_num_tokens):
        """Generate a translated sequence from the source sequence.

        Parameters
        ----------
        x_src : torch.IntTensor
            Tokens of size (1, C_src); B=1. C_src is assumed to be not
            greater than the context window of the encoder.
        x_tgt : torch.IntTensor
            Tokens of size (1, C_tgt); B=1. C_src is assumed to be not
            greater than the context window of the decoder.
        mask : torch.Tensor
        numb_mask : torch.Tensor
        max_num_tokens : int
            Maximum number of tokens to generate. If this value is
            bigger than the context window of the decoder, then we must
            crop the input sequence (to the decoder) to the context
            window at some point.

        Returns
        -------
        

        Notes
        -----
        1. (No mini-batch inference) Currently, we assume B=1 at the
        inference (or generation) phase.
        
        2. (Generate from what) In `forward`, the logits are computed
        for all token positions (i.e., the final linear layer is
        applied to all token positions). In `generate`, on the other
        hand, the final linear and softmax are applied only to the last
        token position of the decoder embeddings.

        3. (Error propagation) During generation (i.e., inference), the
        model uses its own predicted tokens in order to predict the
        next token (vs. teacher forcing during training). Therefore, if
        the model predicts poorly early in the sequence, it is possible
        for the mistake to influence subsequent predictions.
        """
        # Only support B=1.
        B_src, _ = x_src.shape
        B_tgt, _ = x_tgt.shape
        torch._assert(
            (B_src == 1 and B_src == B_tgt), 
            ("Only batch size 1 is allowed during inference, for input "
            "sequences from both src and tgt")
        )

        x_src = archutil.crop_data_to_ctx_window(x_src, self.ctx_window_enc)

        # Constraints on the length of src & tgt sequences
        torch._assert(
            x_src.shape[1] <= self.ctx_window_enc,
            ("Length of input sequence from src must not be greater "
            "than the context window for the encoder")
        )
        torch._assert(
            x_tgt.shape[1] <= self.ctx_window_dec,
            ("Length of input sequence from tgt must not be greater "
            "than the context window for the decoder")
        )

        xs = self.elut(x_src) # (1, C_src, E)
        xs = self.pe_enc(xs) # (1, C_src, E)
        xs = self.encoder(xs, numb_mask) # (1, C_src, E)

        for i in range(max_num_tokens):
            x_tgt = archutil.crop_data_to_ctx_window(x_tgt, self.ctx_window_dec)

            xt = self.elut(x_tgt) # (1, C_tgt, E)
            xt = self.pe_dec(xt) # (1, C_tgt, E)
            xt = self.decoder(xt, xs, mask, numb_mask) # (1, C_tgt, E)

            # Apply the last linear and softmax to the token at the
            # last position only.
            x_last = xt[:, -1: :] # (1, 1, E); B=1; C_tgt=1
            logits = torch.matmul(self.linear, x_last.transpose(-2, -1)) # (V, E) @ (1, E, 1) -> (1, V, 1)
            logits = logits.transpose(-2, -1) # (1, 1, V); B=1; C_tgt=1
            probs = F.softmax(logits, dim=-1) # (1, 1, V); B=1; C_tgt=1
            x_new = gen_token_greedy(probs) # (1, 1); B=1; C_new=1
            if is_eos(x_new):
                break
            x_tgt = torch.cat((x_tgt, x_new), dim=-1) # (1, C_tgt+1)

        return x_tgt
    
    def generate_batch(self, x_src: torch.Tensor, attention_mask_src: torch.Tensor, max_new_tokens: int, sos_token_id: int):
        """
        [배치 처리 버전] 여러 소스 시퀀스로부터 번역 시퀀스를 생성합니다.
        현재 추론 워크로드 효율성을 위하여 배치 처리가 가능하며 현재 활용하는 wmt 데이터셋에 적합한 제너레이트를 만듭니다.
        Parameters
        ----------
        x_src : torch.IntTensor
            소스 토큰 배치. 크기: (B, C_src)
        attention_mask_src : torch.Tensor
            소스 시퀀스의 패딩 마스크. 크기: (B, C_src)
        max_new_tokens : int
            생성할 최대 토큰 수
        sos_token_id : int
            타겟 시퀀스의 시작을 알리는 SOS(Start of Sequence) 토큰 ID
        """
        batch_size = x_src.shape[0]
        
        x_src = archutil.crop_data_to_ctx_window(x_src, self.ctx_window_enc)

        xs = self.elut(x_src)
        xs = self.pe_enc(xs)
        src_seq_len = xs.shape[1]
        enc_padding_mask = torch.zeros(batch_size, src_seq_len, src_seq_len, dtype=torch.bool, device=x_src.device)
        xs = self.encoder(xs, enc_padding_mask) # (B, C_src, E)

        x_tgt = torch.full((batch_size, 1), sos_token_id, dtype=torch.long, device=x_src.device)
        unfinished_sequences = torch.ones(batch_size, dtype=torch.bool, device=x_src.device)

        for _ in range(max_new_tokens):
            x_tgt = archutil.crop_data_to_ctx_window(x_tgt, self.ctx_window_dec)

            seq_len = x_tgt.shape[1]
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x_src.device), diagonal=1).bool()
            causal_mask = causal_mask.unsqueeze(0).expand(batch_size, seq_len, seq_len) 

            # Cross-attention mask (padding)
            src_padding_mask = (attention_mask_src == 0)  # [32, 42]
            cross_attn_mask = src_padding_mask.unsqueeze(2).expand(batch_size, src_seq_len, seq_len)
    
            xt = self.elut(x_tgt)
            xt = self.pe_dec(xt)
            xt = self.decoder(xt, xs, causal_mask, cross_attn_mask)

            # logits = torch.matmul(self.linear, xt[:, -1, :].unsqueeze(-1)) # (B, V, 1)
            # logits = logits.squeeze(-1) # (B, V)

            xt_last = xt[:, -1, :]  # 마지막 토큰 위치
            if xt_last.dim() == 3:  # [32, 42, 512] -> [32, 512]
                xt_last = xt_last[:, -1, :]

            logits = torch.matmul(xt_last, self.linear.T) 
            next_tokens = torch.argmax(logits, dim=-1) # (B,)

            # eos_token_id = self.tokenizer.eos_token_id if hasattr(self, 'tokenizer') else 2
            # next_tokens = next_tokens * unfinished_sequences.long() + eos_token_id * (1 - unfinished_sequences.long())

            x_tgt = torch.cat([x_tgt, next_tokens.unsqueeze(-1)], dim=1)
            # unfinished_sequences = unfinished_sequences & (next_tokens != eos_token_id)
   
            # if not unfinished_sequences.any():
            #     break
        return x_tgt

