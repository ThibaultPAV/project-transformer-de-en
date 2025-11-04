import copy
import math

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0, restore_best_weights=True, best_value = math.inf, best_state = None):
        """
        patience : number of epochs with no improvement before stopping
        min_delta : minimum improvement on the loss
        restore_best_weights : if True, restore best weights at the end

        """
        self.patience = patience
        self.min_delta = float(min_delta)
        self.restore_best_weights = restore_best_weights

        self.best_value = best_value 
        self.best_state = best_state
        self.num_bad_epochs = 0
        self.should_stop = False

    def step(self, current_value, model=None):
        """
        Call after each validation loss computation.
        current_value : current validation loss
        """
        improved = (self.best_value - current_value) > self.min_delta

        if improved:
            self.best_value = current_value
            self.num_bad_epochs = 0
            if self.restore_best_weights and model is not None:
                self.best_state = copy.deepcopy(model.state_dict()) # deepcopy because state_dict changes during training
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self.should_stop = True
        return self.should_stop

    def restore(self, model):
        if self.restore_best_weights and self.best_state is not None:
            model.load_state_dict(self.best_state)
