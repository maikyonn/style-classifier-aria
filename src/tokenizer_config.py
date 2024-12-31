from aria.tokenizer import AbsTokenizer

def get_tokenizer():
    """
    Creates and configures a singleton instance of the AbsTokenizer.
    Returns the same instance on subsequent calls.
    """
    if not hasattr(get_tokenizer, 'tokenizer'):
        tokenizer = AbsTokenizer()
        tokenizer.add_tokens_to_vocab(["A", "B", "C", "D"])
        get_tokenizer.tokenizer = tokenizer
    
    return get_tokenizer.tokenizer

# Common label mappings
LABEL_TO_ID = {"A": 0, "B": 1, "C": 2, "D": 3}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

def get_pad_token():
    """Get the padding token ID."""
    return get_tokenizer().encode(["<P>"])[0] 