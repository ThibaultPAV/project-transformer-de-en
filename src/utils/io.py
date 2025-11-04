from pathlib import Path
import torch, os

def save_checkpoint(path, model, optimizer, scheduler, epoch, global_step, best_bleu, best_val_loss, train_loss, validation_loss, extra_cfg=None):
    """
    Save training checkpoint.
    input:
        path: file path to save the checkpoint
        model: model to save
        optimizer: optimizer to save
        scheduler: learning rate scheduler to save
        epoch: current epoch number
        global_step: current global training step
        best_bleu: best BLEU score achieved so far
        best_val_loss :  best val loss achieved so far
    """
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_bleu": best_bleu,
        "best_val_loss": best_val_loss,
        "validation_loss": validation_loss,
        "train_loss": train_loss,
        "extra": extra_cfg or {},
        "torch_version": str(torch.__version__),
    }
    torch.save(ckpt, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    """
    Load training checkpoint.
    input:
        path: file path to load the checkpoint from
        model: model to load state into
        optimizer: optimizer to load state into (optional)
        scheduler: learning rate scheduler to load state into (optional)
        map_location: device mapping for loading the checkpoint
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("global_step", 0)
    best_bleu = ckpt.get("best_bleu", 0.0)
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    validation_loss = ckpt.get("validation_loss", [])
    train_loss = ckpt.get("train_loss", [])
    return epoch, global_step, best_bleu, best_val_loss, validation_loss, train_loss