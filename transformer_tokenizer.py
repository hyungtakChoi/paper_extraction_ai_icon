"""Tokenizer."""

import torch

def is_eos(x):
    """Is <EOS> (end of sequence).

    Return True if the given token `x` is <EOS>. Return False
    otherwise.

    Parameter
    ---------
    x : torch.Tensor
        Token of size (1, 1); B=1; C=1.

    Returns
    -------
    Boolean
    """
    return False # temporary placeholder # TODO: Implement.

# TODO: Consider using Tokenizer from the HuggingFace Transformers
# library:
#   1. eos, mask, padding, etc.
#   2. Masking in the setup of rectangular attn pattern
