import torch
from .models import XGBoostBaseline, LSTMBaseline, TempCNNBaseline, VanillaTransformer, STSMamba
MODEL_NAMES=('XGBoost','LSTM','TempCNN','VanillaTransformer','STSMamba')
def get_model(model, ndims, num_classes, sequencelength, device, **kwargs):
    kwargs.pop('input_dims',None);kwargs.pop('phenology_indices',None)
    if model=='XGBoost':obj=XGBoostBaseline(input_dim=ndims,num_classes=num_classes,sequencelength=sequencelength,**kwargs)
    elif model=='LSTM':obj=LSTMBaseline(input_dim=ndims,num_classes=num_classes,**kwargs)
    elif model=='TempCNN':obj=TempCNNBaseline(input_dim=ndims,num_classes=num_classes,sequencelength=sequencelength,**kwargs)
    elif model=='VanillaTransformer':obj=VanillaTransformer(input_dim=ndims,num_classes=num_classes,sequencelength=sequencelength,**kwargs)
    elif model=='STSMamba':obj=STSMamba(input_dim=ndims,num_classes=num_classes,seq_len=sequencelength,patch_size=kwargs.pop('spatial_patch_size',3),**kwargs)
    else:raise ValueError(model)
    return obj.to(device)

def save_model_checkpoint(model, path):
    if getattr(model, "is_tree_model", False):
        torch.save({"model_type": "XGBoost", "estimator": model.estimator}, path)
    else:
        torch.save(model.state_dict(), path)


def load_model_checkpoint(model, path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if getattr(model, "is_tree_model", False):
        if not isinstance(checkpoint, dict) or "estimator" not in checkpoint:
            raise ValueError(f"Invalid XGBoost checkpoint: {path}")
        model.estimator = checkpoint["estimator"]
    else:
        model.load_state_dict(checkpoint)
    return model


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, score, model):

        # run "if" for the first batch, run "elif" or "else" after
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        '''Saves model'''
        save_model_checkpoint(model, self.path)
