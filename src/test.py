import torch
import argparse
import evaluate
import matplotlib.pyplot as plt
from pathlib import Path
from tokenizers import Tokenizer
from data.dataset import JsonlMT, pad_batch
from model.model import TinyTransformerMT
from functools import partial
from torch.utils.data import DataLoader
from utils.io import load_checkpoint
from utils.text import detok
from omegaconf import OmegaConf

def test_model(args):

    # Load tokenizer
    device="cpu"
    tok = Tokenizer.from_file(args.tokenizer.file)
    pad_id = tok.token_to_id("<pad>")
    bos_id = tok.token_to_id("<s>")
    eos_id = tok.token_to_id("</s>")
    vocab = tok.get_vocab_size()

    test_ds = JsonlMT(args.test.test_path, args.tokenizer.file, max_len=args.data.max_len)
    collate = partial(pad_batch, pad_id=pad_id)
    test_dl = DataLoader(test_ds, batch_size=args.data.batch_size, shuffle=False, collate_fn=collate)

    # load model
    model = TinyTransformerMT(vocab, d_model=args.model.d_model, nhead=args.model.nhead, num_layers=args.model.layers, dim_ff=args.model.ff, pad_id=pad_id).to(device)
    start_epoch, global_step, best_bleu, best_val_loss, validation_loss, train_loss = load_checkpoint(
    args.eval.model_path, model, optimizer=None, scheduler=None, map_location=device
    )   
    model.eval()
    print(tok.decoder) 
    # plot losses
    plt.figure(figsize=(8,4))
    plt.plot(train_loss, label="Train loss", alpha=0.9)
    plt.xlabel("batches")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.legend()
    plt.show()

    # Graphique 2 : Validation
    plt.figure(figsize=(8,4))
    plt.plot(validation_loss, label="Validation loss", alpha=0.9)
    plt.xlabel("batches")
    plt.ylabel("Loss")
    plt.title("Validation loss")
    plt.legend()
    plt.show()

    #  Evaluation BLEU
    bleu = evaluate.load("sacrebleu")
    preds, refs = [], []
    
    with torch.no_grad():
        for batch in test_dl:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            B = src.size(0)

            ys = torch.full((B,1), bos_id, dtype=torch.long, device=device)
            finished = torch.zeros(B, dtype=torch.bool, device=device)

            for _ in range(args.data.max_len):
                logits = model(src, ys)
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ys = torch.cat([ys, next_tok], dim=1)
                finished |= (next_tok.squeeze(1) == eos_id)
                if finished.all():
                    break

            # Detokenisation
            for seq_ids, ref_ids in zip(ys, tgt):
                preds.append(detok(tok, seq_ids.tolist(), pad_id))
                refs.append([detok(tok, ref_ids.tolist(), pad_id)])

    # Calcul of BLEU score
    result = bleu.compute(predictions=preds, references=refs)
    print(f"\nBleu Score : {result['score']:.2f}")
    
    print(detok(tok, test_ds[0]['src'], pad_id))
    print("\n Exemples of traduction :")
    for i in range(3):
        print(f"\nSRC : {test_ds[i]['src']}")
        print(f"PRED: {preds[i]}")
        print(f"TGT : {refs[i][0]}")


if __name__ == "__main__":
    cfg = OmegaConf.load("configs/config.yaml")
    test_model(cfg)
