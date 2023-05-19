"""Main training loop based on Hydra + Lightening + OmegaConf stack.
Adapted from: https://github.com/HazyResearch/flash-attention
and from: https://github.com/ashleve/lightning-hydra-template
"""
from typing import Optional, List
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import (
    LightningModule,
    LightningDataModule,
    Trainer,
    Callback,
    seed_everything,
)
from pytorch_lightning.loggers.logger import Logger

from src.utils import utils

log = utils.get_logger(__name__)


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

    # init LightningModule for a task
    # the second `cfg=config` argument is used by the task class defined by config.task._target_
    model: LightningModule = hydra.utils.instantiate(config.task, config)
    datamodule: LightningDataModule = model._datamodule

    # init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, callback_cfg in config.callbacks.items():
            if callback_cfg is not None and "_target_" in callback_cfg:
                log.info(f"Instantiating callback <{str(callback_cfg._target_)}>")
                callbacks.append(hydra.utils.instantiate(callback_cfg))

    # init Lightning loggers
    loggers: List[Logger] = []
    if "loggers" in config:
        for _, logger_cfg in config.loggers.items():
            if logger_cfg is not None and "_target_" in logger_cfg:
                log.info(f"Instantiating logger <{str(logger_cfg._target_)}>")
                loggers.append(hydra.utils.instantiate(logger_cfg))

    # resume from checkpoint
    ckpt_cfg = {}
    if config.get("resume"):
        try:
            checkpoint_dir = Path(str(config.callbacks.model_checkpoint.dirpath))
            if checkpoint_dir.is_dir():
                last_ckpt_path = checkpoint_dir / "last.ckpt"
                best_ckpt_path = checkpoint_dir / "best.ckpt"
                if not (last_ckpt_path.exists() or best_ckpt_path.exists()):
                    raise FileNotFoundError("Resume requires either last.ckpt or best.ckpt file to exists.")
                if config.get("resume") == "last":
                    if not last_ckpt_path.exists():
                        log.info("last.ckpt file does not exist, resume from best.ckpt instead.")
                        checkpoint_path = best_ckpt_path
                    else:
                        log.info("Resuming from last.ckpt...")
                        checkpoint_path = last_ckpt_path
                if config.get("resume") == "best":
                    if not best_ckpt_path.exists():
                        log.info("best.ckpt file does not exist, resume from last.ckpt instead...")
                        checkpoint_path = last_ckpt_path
                    else:
                        log.info("Resuming from best.ckpt...")
                        checkpoint_path = best_ckpt_path
                ckpt_cfg = {"ckpt_path": str(checkpoint_path)}
        except (ValueError, FileNotFoundError):
            log.info("Resume failed, trainining from scratch...")

    # init Lightning Trainer
    log.info(f"Instantiating Trainer <{str(config.trainer._target_)}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=loggers)

    # start training
    log.info("Start training!")
    trainer.fit(model=model, datamodule=datamodule, **ckpt_cfg)


