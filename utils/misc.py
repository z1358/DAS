import torch
import logging
from collections import OrderedDict
logger = logging.getLogger(__name__)


@torch.no_grad()
def ema_update_model(model_to_update, model_to_merge, momentum, device, update_all=False):
    if momentum < 1.0:
        for param_to_update, param_to_merge in zip(model_to_update.parameters(), model_to_merge.parameters()):
            if param_to_update.requires_grad or update_all:
                param_to_update.data = momentum * param_to_update.data + (1 - momentum) * param_to_merge.data.to(device)
    return model_to_update

@torch.no_grad()
def ema_update_model_with_name(model_to_update, model_to_merge, momentum, device, name_to_update=None):
    assert name_to_update is not None, "name_to_update should not be None"
    for param_to_update, (name, param_to_merge) in zip(model_to_update.parameters(), model_to_merge.named_parameters()):
        if name_to_update in name:
            update_flag = True
        else:
            update_flag = False
        if update_flag:
            param_to_update.data = momentum * param_to_update.data + (1 - momentum) * param_to_merge.data.to(device)
    return model_to_update

def print_memory_info():
    logger.info('-' * 40)
    mem_dict = {}
    for metric in ['memory_allocated', 'max_memory_allocated', 'memory_reserved', 'max_memory_reserved']:
        mem_dict[metric] = eval(f'torch.cuda.{metric}()')
        logger.info(f"{metric:>20s}: {mem_dict[metric] / 1e6:10.2f}MB")
    logger.info('-' * 40)
    return mem_dict


def rm_substr_from_state_dict(state_dict, substr):
    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        if substr in key:  # to delete prefix 'module.' if it exists
            new_key = key[len(substr):]
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict