import json, os
from tokenizers import Tokenizer, pre_tokenizers, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset
from omegaconf import OmegaConf

"""
Train a BPE tokenizer on the WMT17 German-English dataset.
"""

SPECIALS = ["<pad>", "<s>", "</s>", "<unk>"] #

def stream_texts(dataset): 
    for ex in dataset: 
        t = ex.get("translation", None) # Get translation field
        if t and ("de" in t) and ("en" in t): # Check for both languages
            yield t["de"]; yield t["en"]  # Yield German and English texts -> not stored in memory

def main(args):
    ds = load_dataset("json", data_files={
        "train":"data/raw/train.jsonl",
        "valid":"data/raw/valid.jsonl"
    })

    tok = Tokenizer(BPE(unk_token="<unk>")) # Initialize BPE tokenizer
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)  # Splits on word boundaries and punctuation (not just whitespace)
    tok.decoder = decoders.ByteLevel()
    trainer = BpeTrainer( 
        vocab_size=args.tokenizer.vocab_size,
        special_tokens=SPECIALS,
        min_frequency=2 # because of small dataset
    )

    tok.train_from_iterator(stream_texts(ds["train"]), trainer=trainer) # Train tokenizer on training data

    os.makedirs(args.paths.tokenizer_dir, exist_ok=True) # Create output directory if it doesn't exist
    tok.save(os.path.join(args.paths.tokenizer_dir, "bpe.json")) # Save tokenizer model
    with open(os.path.join(args.paths.tokenizer_dir, "specials.json"), "w") as f: 
        json.dump(SPECIALS, f) # Save special tokens

if __name__ == "__main__":
    cfg = OmegaConf.load("configs/config.yaml")
    main(cfg) # Run main function with parsed arguments
