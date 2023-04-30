
from typing import Dict, List

import torch.nn as nn

from src.models.gpt import GPTLayerNorm
from src.utils.utils import get_logger

logger = get_logger(__file__)


# Adapted from:
# https://github.com/karpathy/nanoGPT/blob/master/model.py
# and
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/optim/param_grouping.py
def group_parameters_for_weight_decay(
    model,
    optim_cfg,
    bias_weight_decay = False,
    normalization_weight_decay = False,
) -> List[Dict]:
    """Group parameters for weight decay.

    Parameters subjected to weight decay:
    - weights in nn.Linear layers
    - biases in nn.Linear layers (if `bias_weight_decay`=True)
    - weights in normalization layers (LayerNorm, BatchNorm) if `normalization_weight_decay`=True

    Parameters subjected to no weight decay:
    - weights and biases in embedding layers (token, position)
    - parameters with `requires_grad=False`
    - parameters set to `no_weight_decay` with markers in the model definition
    - all biases if `bias_weight_decay=False`
    - weights and biases in normalization layers in `normalization_weight_decay=False`
    """
    weight_decay = getattr(optim_cfg, "weight_decay", 0.0)

    # return directly if no weight decay is needed
    if weight_decay == 0.0:
        return model.parameters()

    # add skipped parameters
    # these are defined in each model's own definition
    skip = model.no_weight_decay() if hasattr(model, "no_weight_decay") else set()
    skip_keywords = (model.no_weight_decay_keywords() if hasattr(model, "no_weight_decay_keywords")
                     else set())

    decay = set()
    no_decay = set()
    whitelist = (nn.Linear,)
    blacklist = (nn.Embedding, )
    normalization_modules = (nn.LayerNorm, GPTLayerNorm)
    if not normalization_weight_decay:
        # TODO: how do I add custom layer norm modules?
        blacklist += normalization_modules
    else:
        logger.info("Using weight decay on normalization layers. Make sure this is the desired"
                    "behavior.")
        whitelist += normalization_modules

    # model.named_parameters() won't return duplicates
    params_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    # loop named parameters
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            # full parameter name
            fpn = "%s.%s" % (mn, pn) if mn else pn

            # skip parameters not requiring gradient upate
            # and shared parameters since they won't appear twice in `params_dict`
            if not p.requires_grad or fpn not in params_dict:
                continue

            # add `no_weight_decay` parameters from model
            elif fpn in skip or any(skp in fpn for skp in skip_keywords):
                no_decay.add(fpn)

            # add `no_weight_decay` parameters
            elif hasattr(p, "no_weight_decay"):
                no_decay.add(fpn)

            # biases
            elif pn.endswith("bias"):
                if not bias_weight_decay:
                    no_decay.add(fpn)
                elif isinstance(m, whitelist):
                    decay.add(fpn)

            # weights in whitelist
            elif pn.endswith("weight") and isinstance(m, whitelist):
                decay.add(fpn)

            # blacklist modules
            elif isinstance(m, blacklist):
                no_decay.add(fpn)

    # check that all parameters have been grouped properly
    union_params = decay | no_decay
    inner_params = decay & no_decay
    assert len(inner_params) == 0, (
        f"Parameters {str(inner_params)} were grouped into both decay and no_decay sets."
    )
    assert len(params_dict.keys() - union_params) == 0, (
        f"Parameters {str(params_dict.keys() - union_params)} were not separated into either"
        " decay nor no_decay sets."
    )

    # group parameters
    if weight_decay == 0.0 or not no_decay:
        params_group = [
            {
                "params": [params_dict[pn] for pn in sorted(list(union_params))],
                "weight_decay": weight_decay,
            }
        ]
    else:
        params_group = [
            {
                "params": [params_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [params_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            }
        ]

    return params_group
