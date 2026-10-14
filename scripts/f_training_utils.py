import re
import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
from math import inf
import torch.optim as optim

def update_params(alpha, beta, epoch):
    if epoch <= 4:
        pass
    else:
        alpha -= 0.04
        beta += 0.04
        print(f"New alpha/beta = {alpha, beta} @ epoch {epoch+1}")
    return alpha, beta

def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)

    lr_scheduler = None
    if config.TRAIN.LR_SCHEDULER.NAME == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=(num_steps - warmup_steps) if config.TRAIN.LR_SCHEDULER.WARMUP_PREFIX else num_steps,
            lr_min=config.TRAIN.MIN_LR,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
            warmup_prefix=config.TRAIN.LR_SCHEDULER.WARMUP_PREFIX
        )
    return lr_scheduler

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, enabled=True):
        # enabled=False turns this into a plain backward/clip/step (e.g. AMP off or CPU)
        self._scaler = torch.cuda.amp.GradScaler(enabled=enabled)

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

def build_optimizer(config, model, **kwargs):
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()

    parameters, no_decay_names = set_weight_decay(model, skip, skip_keywords)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None

    # SwinCVS with Multiclassifier (requires E2E=True)
    if config.MODEL.E2E and config.MODEL.MULTICLASSIFIER:
            parameters2= [  {'params': model.swinv2_model.parameters(), 'lr': config.TRAIN.OPTIMIZER.ENCODER_LR},
                            {'params': model.fc_swin.parameters(), 'lr': config.TRAIN.OPTIMIZER.ENCODER_LR},
                            {'params': model.lstm.parameters(), 'lr': config.TRAIN.OPTIMIZER.CLASSIFIER_LR},
                            {'params': model.fc_lstm.parameters(), 'lr': config.TRAIN.OPTIMIZER.CLASSIFIER_LR}]
    # SwinCVS without Multiclassifier
    elif config.MODEL.LSTM:
        parameters2= [  {'params': model.swinv2_model.parameters(), 'lr': config.TRAIN.OPTIMIZER.ENCODER_LR},
                        {'params': model.lstm.parameters(), 'lr': config.TRAIN.OPTIMIZER.CLASSIFIER_LR },
                        {'params': model.fc_lstm.parameters(), 'lr': config.TRAIN.OPTIMIZER.CLASSIFIER_LR}]
    # Bare backbone - swinV2
    else:
        parameters2= [  {'params': model.parameters(), 'lr': config.TRAIN.OPTIMIZER.ENCODER_LR}]

    if opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters2, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
    else:
        raise NotImplementedError

    return optimizer

def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    no_decay_names = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            no_decay_names.append(name)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    

    param_groups = [{'params': has_decay}, {'params': no_decay, 'weight_decay': 0.}]

    return param_groups, no_decay_names 

def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
