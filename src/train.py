import os, argparse, torch, evaluate
import torch.nn as nn
import copy
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from model.earlyStopping import EarlyStopping
from data.dataset import JsonlMT, pad_batch
from model.model import TinyTransformerMT
from model.schedulers import NoamLR
from tqdm import tqdm
from functools import partial
import json, shutil
from pathlib import Path
from utils.io import save_checkpoint, load_checkpoint
from utils.text import detok
from omegaconf import OmegaConf

"""
train a TinyTransformer model on the WMT17 German-English dataset.
"""

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, ignore_index=0):
        """
        Label Smoothing Loss (loss of our model):
            Goal : prevent the model from being too confident about its predictions, which can improve generalization.
        input:
            classes: number of classes (vocabulary size)
            smoothing: smoothing factor for target distribution
            ignore_index: index to ignore in the target
        """
        super().__init__()
        self.kl = nn.KLDivLoss(reduction="batchmean")  # Kullback-Leibler divergence loss
        self.conf = 1.0 - smoothing
        self.smoothing = smoothing
        self.c = classes
        self.ignore = ignore_index

    def forward(self, logits, target):
        """
        input:
            logits: (B, L, V) raw model outputs - B = batch size, L = longueur sequence, V = vocab size
            target: (B, L) target token ids
            exemple:
            logits =
            tensor([[
            [2.0,  0.0, -1.0,  0.5, -0.5],   # word 1
            [1.0, -1.0,  0.0,  2.0,  0.5],   # word 2
            [0.0,  0.0,  0.0,  0.0,  0.0]    # word 3 (padding)
            ]])
            target = tensor([[2, 3, 0]])

        output:
            loss value
        """
        B, L, V = logits.size()
        log_probs = logits.log_softmax(dim=-1) # Because KLDivLoss expects log-probabilities
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs).fill_(self.smoothing/(V-2)) # distribute smoothing
            true_dist.scatter_(-1, target.unsqueeze(-1), self.conf) # put confidence on the target
            true_dist[..., self.ignore] = 0 # zero prob for padding token
            mask = (target == self.ignore) # mask for padding positions
            true_dist[mask] = 0
        return self.kl(log_probs, true_dist)

def main(args):
    device = "cpu" # I dont have an NVIDIA GPU on my current machine
    tok = Tokenizer.from_file(args.tokenizer.file) # Load tokenizer
    pad_id = tok.token_to_id("<pad>")
    bos_id = tok.token_to_id("<s>")
    eos_id = tok.token_to_id("</s>") 
    vocab = tok.get_vocab_size() # Vocabulary size

    train_ds = JsonlMT("data/raw/train.jsonl", args.tokenizer.file, max_len=args.data.max_len)
    valid_ds = JsonlMT("data/raw/valid.jsonl", args.tokenizer.file, max_len=args.data.max_len)

    collate = partial(pad_batch, pad_id=pad_id) # transform for DataLoader by padding
    num_workers = 0 # set to 0 for Windows compatibility

    train_dl = DataLoader(train_ds, batch_size=args.data.batch_size, shuffle=True,
                        collate_fn=collate, num_workers=num_workers,
                        pin_memory=False, persistent_workers=False)
    valid_dl = DataLoader(valid_ds, batch_size=args.data.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=num_workers,
                        pin_memory=False, persistent_workers=False)
    
    model = TinyTransformerMT(vocab, d_model=args.model.d_model, nhead=args.model.nhead, num_layers=args.model.layers,
                              dim_ff=args.model.ff, pad_id=pad_id).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.optim.lr, betas=args.optim.betas, eps=args.optim.eps)
    sched = NoamLR(opt, model_size=args.model.d_model, warmup_steps=args.optim.warmup)
    crit = LabelSmoothingLoss(vocab, smoothing=args.train.label_smoothing, ignore_index=pad_id)

    bleu = evaluate.load("sacrebleu") # for evaluation

    best_bleu = 0.0
    global_step = 0
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = ckpt_dir / "last.pt"
    best_bleu_ckpt = ckpt_dir / "best_bleu.pt"
    best_val_ckpt  = ckpt_dir / "best_val_loss.pt"

    N = args.train.save_every_steps # save parameters every N steps
# ---- Early stopping (validation loss) ----
    
    if last_ckpt.exists():
        start_epoch, global_step, best_bleu, best_val_loss, validation_loss, train_loss = load_checkpoint(
            last_ckpt, model, optimizer=opt, scheduler=sched, map_location=device
        )
        print(f"Resuming from checkpoint {last_ckpt}, epoch {start_epoch}, global step {global_step}, best BLEU {best_bleu:.2f}, best val loss {best_val_loss:.4f}")
    else:
        start_epoch = 1
        best_val_loss = float("inf")
        validation_loss = []
        train_loss = []
    early = EarlyStopping(patience=args.train.early_stopping.patience, min_delta=args.train.early_stopping.min_delta, restore_best_weights=True, best_value = best_val_loss, best_state = copy.deepcopy(model.state_dict()))

    try:      
        for epoch in range(start_epoch, args.train.epochs_max):
            model.train()
            val_loss_sum = 0
            val_tokens = 0
            for batch in tqdm(train_dl, desc=f"Epoch {epoch}"):
                src = batch["src"].to(device)
                tgt = batch["tgt"].to(device)
                y_in  = tgt[:, :-1]   # all except last token
                y_out = tgt[:, 1:]    # all except first token

                logits = model(src, y_in)
                loss   = crit(logits, y_out)

                num_tokens = (y_out != pad_id).sum().item()
                val_loss_sum += loss.item() * num_tokens
                val_tokens   += num_tokens

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient clipping (avoid exploding gradients)
                opt.step()
                sched.step()
                global_step += 1
                train_loss.append(loss.item() * num_tokens)
                print(f"Step {global_step}, Train loss: {(loss.item()):.4f}")
                # save checkpoint every N steps
                if global_step % N == 0:
                    # save last checkpoint
                    save_checkpoint(
                        last_ckpt, model, opt, sched,
                        epoch=epoch, global_step=global_step, best_bleu=best_bleu,
                        best_val_loss=best_val_loss, train_loss=train_loss,
                        validation_loss=validation_loss,
                    )
            print(f"Train loss: {(val_loss_sum / val_tokens):.4f}")

            # Eval simple (greedy) -> its faster 
            model.eval()
            val_loss_sum, val_tokens = 0.0, 0
            preds, refs = [], []

            with torch.no_grad():
                for batch in valid_dl:
                    src = batch["src"].to(device)
                    tgt = batch["tgt"].to(device)

                    # 1) Loss validation, teaching forcing
                    # y_in = shift right, y_out = normal tgt
                    y_in  = tgt[:, :-1]              # remove last token (</s>)
                    y_out = tgt[:, 1:]               # remove first token (<s>)
                    logits = model(src, y_in)       
                    loss = crit(logits, y_out)
                    num_tokens = (y_out != pad_id).sum().item()
                    val_loss_sum += loss.item() * num_tokens
                    val_tokens   += num_tokens

                    # 2) Prediction (greedy)
                    B = src.size(0)
                    ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device) 
                    finished = torch.zeros(B, dtype=torch.bool, device=device)

                    for _ in range(args.data.max_len): # auto-regressive generation -> token
                        logits = model(src, ys) 
                        next_tok = logits[:,-1,:].argmax(dim=-1, keepdim=True) # predict next token
                        ys = torch.cat([ys, next_tok], dim=1) # new sentence with next token added
                        finished |= (next_tok.squeeze(1) == eos_id)
                        if finished.all():
                            break

                    for seq_ids, ref_ids in zip(ys, batch["tgt"]):  # token -> text
                        preds.append(detok(tok, seq_ids, pad_id)) # list of predictions
                        refs.append([detok(tok, ref_ids, pad_id)]) # list of list (sacrebleu format)

            val_loss = val_loss_sum / val_tokens
            validation_loss.append(val_loss)
            print(f"Valid loss: {val_loss:.4f}")

            result = bleu.compute(predictions=preds, references=refs)
            valid_bleu = float(result["score"])
            print(f"Valid BLEU: {valid_bleu:.2f}")

            # Save last checkpoint
            save_checkpoint(
                last_ckpt, model, opt, sched,
                epoch=epoch+1, global_step=global_step, best_bleu=best_bleu,
                best_val_loss=best_val_loss, train_loss=train_loss,
                validation_loss=validation_loss
            )
            # Save best val loss checkpoint if improved
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                shutil.copyfile(last_ckpt, best_val_ckpt)
                print(f"New best val_loss = {best_val_loss:.4f}")

            # Save best checkpoint if improved
            if valid_bleu > best_bleu:
                best_bleu = valid_bleu
                shutil.copyfile(last_ckpt, best_bleu_ckpt)
                print(f"New best BLEU: {best_bleu:.2f}, checkpoint saved.")

            if early.step(val_loss, model=model):
                print(f"Early stopping (patience={early.patience}).")
                early.restore(model)
                save_checkpoint(
                    last_ckpt, model, opt, sched,
                    epoch=epoch+1, global_step=global_step, best_bleu=best_bleu,
                    best_val_loss=best_val_loss, train_loss=train_loss, validation_loss=validation_loss
                )
                shutil.copyfile(last_ckpt, best_val_ckpt)
                break
            


    except KeyboardInterrupt:
        print("Training interrupted. Saving last checkpoint...")
        save_checkpoint(
            last_ckpt, model, opt, sched,
            epoch=epoch, global_step=global_step, best_bleu=best_bleu,
            best_val_loss=best_val_loss, train_loss=train_loss,
            validation_loss=validation_loss
        )



if __name__ == "__main__":
    cfg = OmegaConf.load("configs/config.yaml")
    main(cfg)
