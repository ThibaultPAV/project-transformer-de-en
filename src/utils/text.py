from tokenizers import Tokenizer

def detok(tok, ids, pad_id) :
    ids = [i for i in ids if i != pad_id]
    return tok.decode(ids, skip_special_tokens=True)