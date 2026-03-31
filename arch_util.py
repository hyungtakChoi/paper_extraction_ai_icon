def crop_data_to_ctx_window(x, ctx_window):
    """Crop the data by the context window.

    Crop the data (token sequence) so that the sequence length is less than or
    equal to the context window. It applies to both encoder and decoder. If
    `data` is to the encoder, specify the context window for the encoder; if
    `data` is to the decoder, specify the context window for the decoder.
    """
    if x.shape[1] > ctx_window:
        x = x[:, -ctx_window:]

    return x
