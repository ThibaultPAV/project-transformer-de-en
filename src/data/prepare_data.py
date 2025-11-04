import datasets, os, random
from omegaconf import OmegaConf

"""
Prepare WMT17 German-English dataset by subselecting samples and saving to JSONL files.
"""

def main(args):
    ds = datasets.load_dataset("wmt17", "de-en") # Load WMT17 German-English dataset

    def subselect(split, n): # Subselect n samples from the specified split
        n = min(n, len(ds[split])) # Ensure n does not exceed dataset size
        idx = list(range(len(ds[split]))) 
        random.seed(args.seed); random.shuffle(idx) # Ensure reproducibility
        return ds[split].select(idx[:n]) # Select first n shuffled samples

    train = subselect("train", args.data.train_samples) # Subselect training samples
    valid = ds["validation"]
    test  = ds["test"]

    os.makedirs(args.paths.data_raw, exist_ok=True) # Create output directory if it doesn't exist
    train.to_json(os.path.join(args.paths.data_raw, "train.jsonl"))
    valid.to_json(os.path.join(args.paths.data_raw, "valid.jsonl"))
    test.to_json(os.path.join(args.paths.data_raw, "test.jsonl"))

if __name__ == "__main__":
    cfg = OmegaConf.load("configs/config.yaml")
    main(cfg) 


