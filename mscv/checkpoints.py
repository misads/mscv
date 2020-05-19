import torch
import os
import os.path as osp
from collections import OrderedDict
import torch.nn as nn
import misc_utils as utils


def load_state_dict(module, state_dict):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.

    """
    try:
        module.load_state_dict(state_dict)
    except:
        try:
            model_dict = module.state_dict()
            not_initialized = {k.split('.')[0] for k, v in state_dict.items() if k not in model_dict}
            state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            module.load_state_dict(state_dict)
            utils.color_print('Warning: Pretrained network has excessive layers: ', 1, end='')
            utils.color_print(str(sorted(not_initialized)), 1)
        except:
            utils.color_print('Warning: Pretrained network has fewer layers; The following are not initialized: ', 1, end='')
            for k, v in state_dict.items():
                if v.size() == model_dict[k].size():
                    model_dict[k] = v

            not_initialized = set()

            for k, v in model_dict.items():
                if k not in state_dict or v.size() != state_dict[k].size():
                    not_initialized.add(k.split('.')[0])

            utils.color_print(str(sorted(not_initialized)), 1)
            module.load_state_dict(model_dict)


def load_checkpoint(load_dict,
                    filename,
                    map_location=None,
                    strict=False,
                    logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Either a filepath or URL or modelzoo://xxxxxxx.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    state_dict = torch.load(filename, map_location)
    # get state_dict from checkpoint

    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    # load state_dict

    for key, model in load_dict.items():
        if hasattr(model, 'module'):
            load_state_dict(model.module, state_dict[key])
        else:
            load_state_dict(model, state_dict[key])

    return state_dict


def save_checkpoint(save_dict, filename):
    """Save checkpoint to file.

    The checkpoint will have 3 fields: ``meta``, ``state_dict`` and
    ``optimizer``.

    Args:
        save_dict (dict): string to module map.
        filename (str): Checkpoint filename.
    """
    os.makedirs(osp.dirname(filename), exist_ok=True)

    for key in save_dict:
        model = save_dict[key]
        if isinstance(model, nn.Module):
            if hasattr(model, 'module'):
                save_dict[key] = model.module

        if hasattr(save_dict[key], 'state_dict'):
            save_dict[key] = save_dict[key].state_dict()

    torch.save(save_dict, filename)

