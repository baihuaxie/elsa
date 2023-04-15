"""Main training loop based on Hydra + Lightening + OmegaConf stack.
Adapted from: https://github.com/HazyResearch/flash-attention
and from: https://github.com/ashleve/lightning-hydra-template
"""
from typing import Optional
import pyrootutils

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import (
    LightningModule,
    LightningDataModule,
    seed_everything,
)

# this adds project root directory to PYTHONPATH
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import utils

logger = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Instantiate all PyTorch Lightning objects from `config`.

    Args:
        config (DictConfig): Configuration object composed by Hydra.
    
    Returns:
        Optional[float]: Metric scores for hyperparameter search.
    """
    # set random seeds for pytorch, cuda, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # set to False to allow adding new fields to `config` w/o reporint errors
    # note: this should be the default behavior for OmegaConf, but just to be sure
    # ref: https://omegaconf.readthedocs.io/en/2.0_branch/usage.html#access-and-manipulation
    OmegaConf.set_struct(config, False)

    # init Lightning model and datamodule for a task
    # the second `cfg=config` argument is used by the model class defined by config.task._target_
    model: LightningModule = hydra.utils.instantiate(config.task, cfg=config)
    datamodule: LightningDataModule = model.datamodule
