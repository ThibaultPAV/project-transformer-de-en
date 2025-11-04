from torch.optim.lr_scheduler import _LRScheduler
#Noam learning rate scheduler as described in "Attention is all you need" paper
#Goal : increase the learning rate linearly for the first warmup_steps training steps, then decrease it proportionally to the inverse square root of the step number.

class NoamLR(_LRScheduler):
    """
    Noam Learning Rate Scheduler, heriting from _LRScheduler.
    """
    def __init__(self, optimizer, model_size, warmup_steps=4000, last_epoch=-1):
        self.model_size = model_size
        self.warmup = warmup_steps
        super().__init__(optimizer, last_epoch) 

    def get_lr(self): # Compute learning rate at current step
        step = max(1, self._step_count) # step count starts at 1
        scale = (self.model_size ** -0.5) * min(step ** -0.5, step * (self.warmup ** -1.5))
        return [base_lr * scale for base_lr in self.base_lrs]  #list of learning rates for each param group
