import enum
import importlib
from dataclasses import dataclass
from typing import Optional, List
import textwrap
import random

### TODO
### 1. Double-check None's (config attributes)

MULTIPLE_OF_FOR_D_FF = 32

class ArchType(enum.Enum):
    TRANSFORMER = enum.auto()
    BERT = enum.auto()
    GPT1 = enum.auto()
    GPT2 = enum.auto()
    GPT3 = enum.auto()
    LLAMA1 = enum.auto()
    LLAMA2 = enum.auto()
    LLAMA3 = enum.auto()

    @staticmethod
    def get_module(archtype):
        return importlib.import_module(".".join(["arch", archtype.name.lower()]))

class DataTypeTransformer(enum.Enum):
    WMT_EN_DE = enum.auto()
    WMT_EN_FR = enum.auto()

    @classmethod
    def randomly(cls):
        return random.choice(list(DataTypeTransformer))

class DataTypeBert(enum.Enum):
    MNLI = enum.auto()
    QQP = enum.auto()
    QNLI = enum.auto()
    SST2 = enum.auto()
    COLA = enum.auto()
    STSB = enum.auto()
    MRPC = enum.auto()
    RTE = enum.auto()
    
    @classmethod
    def randomly(cls):
        return random.choice(list(DataTypeBert))

class DataTypeGpt(enum.Enum):
    MMLU = enum.auto()
    TRIVIAQA = enum.auto()
    NQ = enum.auto()
    GSM8K = enum.auto()

    @classmethod
    def randomly(cls):
        return random.choice(list(DataTypeGpt))

class DataTypeLlama(enum.Enum):
    MMLU = enum.auto()
    TRIVIAQA = enum.auto()
    NQ = enum.auto()
    GSM8K = enum.auto()

    @classmethod
    def randomly(cls):
        return random.choice(list(DataTypeLlama))

@dataclass(frozen=False)
class Config:
    """Transformer config."""

    # RFE: one TFConfig class per problem? (e.g., NMT, NTP)
    # REF: more generalizatoin? (e.g., one vocab. vs. two vocabs.)
    # REF: more configs?
    #   - type of activation: relu, gelu, tanh, sigmoid, ...
    #   - sampling strategy: greedy decoding, multinomial sampling, ...
    #   - temperature

    # problem setting
    vocab_size: int # (V)

    # data
    batch_size: int
    ctx_window_enc: int # max sequence length that the encoder may take
    ctx_window_dec: int # max sequence length that the decoder may take
    d_emb: int # (E)
    d_q: int
    d_k: int
    d_v: int # for factors of W_V (low-rank transformation)
    d_ff: int

    # architecture
    n_heads_enc: int # number of heads in encoder
    n_heads_dec_sa: int # number of heads in cross-attention in decoder
    n_heads_dec_ca: int # number of heads in self-attention in decoder
    n_layers_enc: int
    n_layers_dec: int
    dropout_rate_enc: float
    dropout_rate_dec: float

    n_groups: Optional[int] = None

    sparse_type: Optional[str] = None
    sparse_factor: Optional[int] = None
    sparse_layers: Optional[List[int]] = None

    # ==================================
    # Architecture-specific constructors
    # ==================================

    @classmethod
    def for_transformer(
        cls,
        batch_size, ctx_window_enc, ctx_window_dec, d_qkv,
        n_heads_enc_dec, n_layers_enc_dec, dropout_rate_enc_dec):

        d_emb = d_qkv * n_heads_enc_dec
        d_ff = d_emb * 4

        return cls(
            vocab_size=None, # NOTE: data/task-specific; the right value is supposed to be assigned before inference.
            batch_size=batch_size,
            ctx_window_enc=ctx_window_enc,
            ctx_window_dec=ctx_window_dec,
            d_emb=d_emb,
            d_q=d_qkv,
            d_k=d_qkv,
            d_v=None,
            d_ff=d_ff,
            n_heads_enc=n_heads_enc_dec,
            n_heads_dec_sa=n_heads_enc_dec,
            n_heads_dec_ca=n_heads_enc_dec,
            n_layers_enc=n_layers_enc_dec,
            n_layers_dec=n_layers_enc_dec,
            dropout_rate_enc=dropout_rate_enc_dec,
            dropout_rate_dec=dropout_rate_enc_dec
        )

    @classmethod
    def for_bert(
        cls,
        batch_size, ctx_window_enc, d_qkv,
        n_heads_enc, n_layers_enc, dropout_rate_enc):

        d_emb = d_qkv * n_heads_enc
        d_ff = d_emb * 4

        return cls(
            vocab_size=30522,
            batch_size=batch_size,
            ctx_window_enc=ctx_window_enc,
            ctx_window_dec=None,
            d_emb=d_emb,
            d_q=d_qkv,
            d_k=d_qkv,
            d_v=d_qkv,
            d_ff=d_ff,
            n_heads_enc=n_heads_enc,
            n_heads_dec_sa=None,
            n_heads_dec_ca=None,
            n_layers_enc=n_layers_enc,
            n_layers_dec=None,
            dropout_rate_enc=dropout_rate_enc,
            dropout_rate_dec=None
        )

    @classmethod
    def for_gpt1(
        cls,
        batch_size, ctx_window_dec, d_qkv,
        n_heads_dec, n_layers_dec, dropout_rate_dec):

        d_emb = d_qkv * n_heads_dec
        d_ff = d_emb * 4

        return cls(
            vocab_size=40478,
            batch_size=batch_size,
            ctx_window_enc=None,
            ctx_window_dec=ctx_window_dec,
            d_emb=d_emb,
            d_q=d_qkv,
            d_k=d_qkv,
            d_v=d_qkv,
            d_ff=d_ff,
            n_heads_enc=None,
            n_heads_dec_sa=n_heads_dec,
            n_heads_dec_ca=None,
            n_layers_enc=None,
            n_layers_dec=n_layers_dec,
            dropout_rate_enc=None,
            dropout_rate_dec=dropout_rate_dec
        )

    @classmethod
    def for_gpt2(
        cls,
        batch_size, ctx_window_dec, d_qkv,
        n_heads_dec, n_layers_dec, dropout_rate_dec):

        d_emb = d_qkv * n_heads_dec
        d_ff = d_emb * 4

        return cls(
            vocab_size=50257,
            batch_size=batch_size,
            ctx_window_enc=None,
            ctx_window_dec=ctx_window_dec,
            d_emb=d_emb,
            d_q=d_qkv,
            d_k=d_qkv,
            d_v=d_qkv,
            d_ff=d_ff,
            n_heads_enc=None,
            n_heads_dec_sa=n_heads_dec,
            n_heads_dec_ca=None,
            n_layers_enc=None,
            n_layers_dec=n_layers_dec,
            dropout_rate_enc=None,
            dropout_rate_dec=dropout_rate_dec
        )

    @classmethod
    def for_gpt3(
        cls,
        batch_size, ctx_window_dec, d_qkv,
        n_heads_dec, n_layers_dec, dropout_rate_dec):

        d_emb = d_qkv * n_heads_dec
        d_ff = d_emb * 4

        return cls(
            vocab_size=50257,
            batch_size=batch_size,
            ctx_window_enc=None,
            ctx_window_dec=ctx_window_dec,
            d_emb=d_emb,
            d_q=d_qkv,
            d_k=d_qkv,
            d_v=d_qkv,
            d_ff=d_ff,
            n_heads_enc=None,
            n_heads_dec_sa=n_heads_dec,
            n_heads_dec_ca=None,
            n_layers_enc=None,
            n_layers_dec=n_layers_dec,
            dropout_rate_enc=None,
            dropout_rate_dec=dropout_rate_dec
        )

    @classmethod
    def for_llama1(
        cls,
        batch_size, ctx_window_dec, d_qkv, d_ff,
        n_heads_dec, n_layers_dec, dropout_rate_dec):

        d_emb = d_qkv * n_heads_dec

        assert d_emb * 8 // 3 <= d_ff
        assert d_ff <= d_emb * 4
        assert d_ff % MULTIPLE_OF_FOR_D_FF == 0

        return cls(
            vocab_size=32000,
            batch_size=batch_size,
            ctx_window_enc=None,
            ctx_window_dec=ctx_window_dec,
            d_emb=d_emb,
            d_q=d_qkv,
            d_k=d_qkv,
            d_v=d_qkv,
            d_ff=d_ff,
            n_heads_enc=None,
            n_heads_dec_sa=n_heads_dec,
            n_heads_dec_ca=None,
            n_layers_enc=None,
            n_layers_dec=n_layers_dec,
            dropout_rate_enc=None,
            dropout_rate_dec=dropout_rate_dec
        )

    @classmethod
    def for_llama2(
        cls,
        batch_size, ctx_window_dec, d_qkv, d_ff,
        n_heads_dec, n_layers_dec, dropout_rate_dec):

        d_emb = d_qkv * n_heads_dec

        assert d_emb * 8 // 3 <= d_ff
        assert d_ff <= d_emb * 4
        assert d_ff % MULTIPLE_OF_FOR_D_FF == 0

        return cls(
            vocab_size=32000,
            batch_size=batch_size,
            ctx_window_enc=None,
            ctx_window_dec=ctx_window_dec,
            d_emb=d_emb,
            d_q=d_qkv,
            d_k=d_qkv,
            d_v=d_qkv,
            d_ff=d_ff,
            n_heads_enc=None,
            n_heads_dec_sa=n_heads_dec,
            n_heads_dec_ca=None,
            n_layers_enc=None,
            n_layers_dec=n_layers_dec,
            dropout_rate_enc=None,
            dropout_rate_dec=dropout_rate_dec
        )

    @classmethod
    def for_llama3(
        cls,
        batch_size, ctx_window_dec, d_qkv, d_ff,
        n_heads_dec, n_layers_dec, dropout_rate_dec):

        d_emb = d_qkv * n_heads_dec

        assert d_emb * 8 // 3 <= d_ff
        assert d_ff <= d_emb * 4
        assert d_ff % MULTIPLE_OF_FOR_D_FF == 0

        return cls(
            vocab_size=128256,
            batch_size=batch_size,
            ctx_window_enc=None,
            ctx_window_dec=ctx_window_dec,
            d_emb=d_emb,
            d_q=d_qkv,
            d_k=d_qkv,
            d_v=d_qkv,
            d_ff=d_ff,
            n_heads_enc=None,
            n_heads_dec_sa=n_heads_dec,
            n_heads_dec_ca=None,
            n_layers_enc=None,
            n_layers_dec=n_layers_dec,
            dropout_rate_enc=None,
            dropout_rate_dec=dropout_rate_dec
        )

    # =================================
    # Random attribute-value generators
    # =================================

    @staticmethod
    def _gen_random_batch_size():
        return random.choice([1, 2, 4])

    @staticmethod
    def _gen_random_ctx_window_enc():
        # public configs: bert=512, gpt1=512, gpt2=1024, gpt3=2048, llama1=2048(=2^11), llama2=4096(=2^12), llama3=8192(=2^13)
        return random.randint(2**9, 2**13) # {2^9=512, 513, 514, ..., 8190, 8191, 2^{13}=8192}; all integers within the range

    @staticmethod
    def _gen_random_ctx_window_dec():
        return random.randint(2**9, 2**13)

    @staticmethod
    def _gen_random_d_qkv():
        # public configs: transformer=64, bert={64, 256}, gpt={64, 96, 128}, llama=128
        return 2 * random.randint(2**5//2, 2**9//2) # {2^5=32, 34, ..., 510, 2^9=512}; only even numbers within the range

    @staticmethod
    def _gen_random_d_ff_llama(d_emb):
        # RFE: can optimize the computation by only considering multiples
        # of `MULTIPLE_OF_FOR_D_FF`, instead of checking all the integers in the range.
        # We use supoptimal computation due to simplicity.
        choices = [num for num in range(d_emb*8//3, d_emb*4+1) if num % MULTIPLE_OF_FOR_D_FF == 0]
        try:
            return random.choice(choices)
        except IndexError as e:
            print(f"no number to sample: {e}")

    @staticmethod
    def _gen_random_n_heads():
        # public configs: transformer={8, 16}, bert={2, 3, 4, 8, 12, 16}, gpt={12, 16, 20, 24, 25, 32, 40, 96}, llama={32, 40, 52, 64}
        return random.randint(1, 100) # {1, 2, ..., 100}

    @staticmethod
    def _gen_random_n_layers():
        # public configs: transformer=6, bert={2, 3, 4, 6, 8, 12, 24}, gpt={12, 24, 32, 36, 40, 48, 96}, llama={32, 40, 48, 60, 80}
        return random.randint(1, 100) # {1, 2, ..., 100}

    @staticmethod
    def _gen_random_dropout_rate():
        return random.choice([.1, .2, .3, .4, .5])

    # =========================================
    # Architecture-specific random constructors
    # =========================================

    ### Constraints enforced during generation:
    ### 1. d_q = d_k = d_v								--> Done
    ### 2. d_emb = d_qkv * n_heads (the kind of heads depends on archtype)		--> Done
    ###   - (step 1) rand-gen d_qkv.
    ###   - (step 2) rand-gen n_heads.
    ###   - (step 3) d_emb = d_q * n_heads
    ### 2-1. (transformer) n_heads for encoder and decoder are identical		--> Done
    ### 3-1. (transformer, bert, gpt) d_ff = d_emb * 4					--> Done
    ### 3-2. (llama) d_ff = a multiple of 256 randomly sampled in [d_emb*8/3, d_emb*4]	--> Done
    ### 4. (transformer) set `vocab_size=None` (data/task-dependent)			--> Done
    ### 5. (transformer) n_layers for encoder and decoder are identical			--> Done
    ### 6. (bert, gpt, llama) set `vocab_size` to the right constant			--> Done
    ### 7. (transformer) dropout_rates for encoder and decoder are identical		--> Done
    
    # @classmethod
    # def for_transformer_random(cls):
    #     return cls.for_transformer(
    #         batch_size=Config._gen_random_batch_size(),
    #         ctx_window_enc=Config._gen_random_ctx_window_enc(),
    #         ctx_window_dec=Config._gen_random_ctx_window_dec(),
    #         d_qkv=Config._gen_random_d_qkv(),
    #         n_heads_enc_dec=Config._gen_random_n_heads(),
    #         n_layers_enc_dec=Config._gen_random_n_layers(),
    #         dropout_rate_enc_dec=Config._gen_random_dropout_rate(),
    #     )

    @classmethod
    def for_transformer_random(cls):
        batch_size = Config._gen_random_batch_size()
        ctx_enc = random.randint(2**7, 2**9) # {2^7=128, 129, 130, ..., 510, 511, 2^{10}=512}; all integers within the range
        ctx_dec = random.randint(2**7, 2**9)
        d_qkv = random.choice([32, 64, 96, 128]) # bert 64 256
        n_heads = random.randint(1, 8) * 2 # random([2, 4, 6, 8, 10, 12, 14, 16]) # 2 ~ 16
        n_layers = random.randint(1, 24)
        dropout = Config._gen_random_dropout_rate()
        return cls.for_transformer(batch_size, ctx_enc, ctx_dec, d_qkv, n_heads, n_layers, dropout)

    # @classmethod
    # def for_bert_random(cls):
    #     return cls.for_bert(
    #         batch_size=Config._gen_random_batch_size(),
    #         ctx_window_enc=Config._gen_random_ctx_window_enc(),
    #         d_qkv=Config._gen_random_d_qkv(),
    #         n_heads_enc=Config._gen_random_n_heads(),
    #         n_layers_enc=Config._gen_random_n_layers(),
    #         dropout_rate_enc=Config._gen_random_dropout_rate()
    #     )

    @classmethod
    def for_bert_random(cls):
        batch_size = Config._gen_random_batch_size()#random.choice([8, 16, 32, 64])
        # ctx_enc = random.choice([128, 256, 384, 512])
        ctx_enc = random.randint(2**7, 2**9) # {2^7=128, 129, 130, ..., 510, 511, 2^{10}=512}; all integers within the range
        d_qkv = random.choice([32, 64, 96, 128, 192, 256])
        n_heads = random.randint(1, 12) * 2 # 2 ~24
        n_layers = random.randint(1, 24) 
        dropout = Config._gen_random_dropout_rate()
        return cls.for_bert(batch_size, ctx_enc, d_qkv, n_heads, n_layers, dropout)

    # @classmethod
    # def for_gpt1_random(cls):
    #     return cls.for_gpt1(
    #         batch_size=Config._gen_random_batch_size(),
    #         ctx_window_dec=Config._gen_random_ctx_window_dec(),
    #         d_qkv=Config._gen_random_d_qkv(),
    #         n_heads_dec=Config._gen_random_n_heads(),
    #         n_layers_dec=Config._gen_random_n_layers(),
    #         dropout_rate_dec=Config._gen_random_dropout_rate()
    #     )

    @classmethod
    def for_gpt1_random(cls):
        batch_size = Config._gen_random_batch_size()#random.choice([2, 4, 8])
        ctx_dec = random.randint(2**8, 2**9) # {2^8=256, 257, 258, ..., 510, 511, 2^{9}=512}; all integers within the range
        d_qkv = random.choice([32, 64, 96])
        n_heads = random.randint(1, 8) * 2
        n_layers = random.randint(1, 15) 
        dropout = Config._gen_random_dropout_rate()
        return cls.for_gpt1(batch_size, ctx_dec, d_qkv, n_heads, n_layers, dropout)

    # @classmethod
    # def for_gpt2_random(cls):
    #     return cls.for_gpt2(
    #         batch_size=Config._gen_random_batch_size(),
    #         ctx_window_dec=Config._gen_random_ctx_window_dec(),
    #         d_qkv=Config._gen_random_d_qkv(),
    #         n_heads_dec=Config._gen_random_n_heads(),
    #         n_layers_dec=Config._gen_random_n_layers(),
    #         dropout_rate_dec=Config._gen_random_dropout_rate()
    #     )

    @classmethod
    def for_gpt2_random(cls):
        batch_size = Config._gen_random_batch_size()#random.choice([1, 2, 4])
        ctx_dec = random.randint(2**9, 2**10) # {2^9=512, 513, 514, ..., 1022, 1023, 2^{10}=1024}; all integers within the range
        d_qkv = random.choice([64, 96, 128])
        n_heads = random.randint(1, 32) * 2
        n_layers = random.randint(1, 48) 
        dropout = Config._gen_random_dropout_rate()
        return cls.for_gpt2(batch_size, ctx_dec, d_qkv, n_heads, n_layers, dropout)

    # @classmethod
    # def for_gpt3_random(cls):
    #     return cls.for_gpt3(
    #         batch_size=Config._gen_random_batch_size(),
    #         ctx_window_dec=Config._gen_random_ctx_window_dec(),
    #         d_qkv=Config._gen_random_d_qkv(),
    #         n_heads_dec=Config._gen_random_n_heads(),
    #         n_layers_dec=Config._gen_random_n_layers(),
    #         dropout_rate_dec=Config._gen_random_dropout_rate()
    #     )

    @classmethod
    def for_gpt3_random(cls):
        batch_size = Config._gen_random_batch_size()#random.choice([1, 2, 4])
        ctx_dec = random.randint(2**9, 2**11) # {2^9=512, 513, 514, ..., 2046, 2047, 2^{12}=2048}; all integers within the range
        d_qkv = random.choice([64, 96, 128])
        n_heads = random.randint(1, 48) * 2
        n_layers = random.randint(1, 96) 
        dropout = Config._gen_random_dropout_rate()
        return cls.for_gpt3(batch_size, ctx_dec, d_qkv, n_heads, n_layers, dropout)

    # @classmethod
    # def for_llama1_random(cls):
    #     d_qkv = Config._gen_random_d_qkv()
    #     n_heads_dec = Config._gen_random_n_heads()
    #     d_emb = d_qkv * n_heads_dec

    #     return cls.for_llama1(
    #         batch_size=Config._gen_random_batch_size(),
    #         ctx_window_dec=Config._gen_random_ctx_window_dec(),
    #         d_qkv=d_qkv,
    #         d_ff=Config._gen_random_d_ff_llama(d_emb),
    #         n_heads_dec=n_heads_dec,
    #         n_layers_dec=Config._gen_random_n_layers(),
    #         dropout_rate_dec=Config._gen_random_dropout_rate()
    #     )

    @classmethod
    def for_llama1_random(cls):
        d_qkv = random.choice([32, 64, 96])
        n_heads_dec = random.randint(1, 16) * 2 # 2 ~ 32
        d_emb = d_qkv * n_heads_dec

        return cls.for_llama1(
            batch_size=Config._gen_random_batch_size(),#random.choice([2, 4, 8]),
            ctx_window_dec= random.randint(2**9, 2**11), # {2^9=512, 513, 514, ..., 2046, 2047, 2^{11}=2048}; all integers within the range
            d_qkv=d_qkv,
            d_ff=Config._gen_random_d_ff_llama(d_emb),
            n_heads_dec=n_heads_dec,
            n_layers_dec=random.randint(1, 40) ,
            dropout_rate_dec=Config._gen_random_dropout_rate()
        )

    # @classmethod
    # def for_llama2_random(cls):
    #     d_qkv = Config._gen_random_d_qkv()
    #     n_heads_dec = Config._gen_random_n_heads()
    #     d_emb = d_qkv * n_heads_dec

    #     return cls.for_llama2(
    #         batch_size=Config._gen_random_batch_size(),
    #         ctx_window_dec=Config._gen_random_ctx_window_dec(),
    #         d_qkv=d_qkv,
    #         d_ff=Config._gen_random_d_ff_llama(d_emb),
    #         n_heads_dec=n_heads_dec,
    #         n_layers_dec=Config._gen_random_n_layers(),
    #         dropout_rate_dec=Config._gen_random_dropout_rate()
    #     )

    @classmethod
    def for_llama2_random(cls):
        d_qkv = random.choice([64, 96, 128])
        n_heads_dec = random.randint(1, 24) * 2 # 2 ~ 48
        d_emb = d_qkv * n_heads_dec

        return cls.for_llama2(
            batch_size=Config._gen_random_batch_size(),#random.choice([1, 2, 4]),
            ctx_window_dec= random.randint(2**9, 2**12), # {2^9=512, 513, 514, ..., 4094, 4095, 2^{12}=4096}; all integers within the range
            d_qkv=d_qkv,
            d_ff=Config._gen_random_d_ff_llama(d_emb),
            n_heads_dec=n_heads_dec,
            n_layers_dec=random.randint(1, 60),
            dropout_rate_dec=Config._gen_random_dropout_rate()
        )

    # @classmethod
    # def for_llama3_random(cls):
    #     d_qkv = Config._gen_random_d_qkv()
    #     n_heads_dec = Config._gen_random_n_heads()
    #     d_emb = d_qkv * n_heads_dec

    #     return cls.for_llama3(
    #         batch_size=Config._gen_random_batch_size(),
    #         ctx_window_dec=Config._gen_random_ctx_window_dec(),
    #         d_qkv=d_qkv,
    #         d_ff=Config._gen_random_d_ff_llama(d_emb),
    #         n_heads_dec=n_heads_dec,
    #         n_layers_dec=Config._gen_random_n_layers(),
    #         dropout_rate_dec=Config._gen_random_dropout_rate()
    #     )

    @classmethod
    def for_llama3_random(cls):
        d_qkv = random.choice([64, 96, 128])
        n_heads_dec = random.randint(1, 32) * 2 # 2 ~ 64
        d_emb = d_qkv * n_heads_dec

        return cls.for_llama3(
            batch_size=Config._gen_random_batch_size(),#random.choice([1, 2, 4]),
            ctx_window_dec=random.randint(2**9, 2**13), # {2^9=512, 513, 514, ..., 8190, 8191, 2^{13}=8192}; all integers within the range
            d_qkv=d_qkv,
            d_ff=Config._gen_random_d_ff_llama(d_emb),
            n_heads_dec=n_heads_dec,
            n_layers_dec=random.randint(1, 80),
            dropout_rate_dec=Config._gen_random_dropout_rate()
        )

    @classmethod
    def randomly(cls, archtype):
        
        def get_constructor(archtype):
            return getattr(Config, "_".join(["for", archtype.name.lower(), "random"]))

        return get_constructor(archtype)()

    # ======================================
    # Instantiation code, randomly generated
    # ======================================

    @staticmethod
    def _get_str_instantiating_config(config):
        return textwrap.dedent(f'''\
            config = Config(
                vocab_size={config.vocab_size},
                batch_size={config.batch_size},
                ctx_window_enc={config.ctx_window_enc},
                ctx_window_dec={config.ctx_window_dec},
                d_emb={config.d_emb},
                d_q={config.d_q},
                d_k={config.d_k},
                d_v={config.d_v},
                d_ff={config.d_ff},
                n_heads_enc={config.n_heads_enc},
                n_heads_dec_sa={config.n_heads_dec_sa},
                n_heads_dec_ca={config.n_heads_dec_ca},
                n_layers_enc={config.n_layers_enc},
                n_layers_dec={config.n_layers_dec},
                dropout_rate_enc={config.dropout_rate_enc},
                dropout_rate_dec={config.dropout_rate_dec}
            )
            '''
        )

    @staticmethod
    def get_code_instantiating_random(archtype):

        config = Config.randomly(archtype)
        return Config._get_str_instantiating_config(config)

