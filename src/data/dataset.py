import json, torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

"""
Dataset class for Machine Translation using JSONL files and a BPE tokenizer."""

class JsonlMT(Dataset): 

    def __init__(self, path, tokenizer_path, src_lang="de", tgt_lang="en", max_len=128):
        self.rows = [json.loads(l) for l in open(path, "r", encoding="utf-8")] # Load all rows from JSONL file
        self.tok = Tokenizer.from_file(tokenizer_path) # Load tokenizer from file
        self.max_len = max_len # Maximum sequence length
        self.src, self.tgt = src_lang, tgt_lang # Source and target language codes
        self.pad_id = self.tok.token_to_id("<pad>") # Padding token ID
        self.bos_id = self.tok.token_to_id("<s>") # Beginning-of-sequence token ID
        self.eos_id = self.tok.token_to_id("</s>") # End-of-sequence token ID

    def encode(self, text):
        ids = self.tok.encode(text).ids[: self.max_len-2] # truncate to max_len minus 2 for BOS and EOS
        return [self.bos_id] + ids + [self.eos_id] # Add BOS and EOS tokens

    def __getitem__(self, i):
        t = self.rows[i]["translation"] # Get translation dictionary
        return {"src": self.encode(t[self.src]), "tgt": self.encode(t[self.tgt])}

    def __len__(self): return len(self.rows)

def pad_batch(examples, pad_id):
    """
    Pad a batch of examples to the same length.

    input:
        examples: list of dicts with "src" and "tgt" keys
        pad_id: token ID used for padding
        exemple : examples = [
        {"src": [11, 22, 33, 2], "tgt": [101, 102, 2]},
        {"src": [44, 55, ,2], "tgt": [103, 104, 105, 2]},
        {"src": [66, 77, 88, 99, 2], "tgt": [106, 2]}
        ]
        pad_id = 0 

        output:
        tensors with padded sequences
        exemple output:
        {
            "src": tensor([[11, 22, 33, 2,  0],
                           [44, 55, 2, 0,  0],
                           [66, 77, 88, 99, 2]]),
            "tgt": tensor([[101, 102, 2,   0],
                           [103, 104, 105, 2],
                           [106, 2,   0,   0]])
        }
    """
    max_src = max(len(x["src"]) for x in examples) # find max source length
    max_tgt = max(len(x["tgt"]) for x in examples)
    src = [x["src"] + [pad_id]*(max_src-len(x["src"])) for x in examples] # add padding 
    tgt = [x["tgt"] + [pad_id]*(max_tgt-len(x["tgt"])) for x in examples]
    return {
        "src": torch.tensor(src, dtype=torch.long), # convert to tensors
        "tgt": torch.tensor(tgt, dtype=torch.long),
    }



