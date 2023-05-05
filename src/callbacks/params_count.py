import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import Callback
from omegaconf import DictConfig


class ParamsCounter(Callback):
    """Count model parameters.
    """
    def __init__(
        self,
        log_total_params: bool = True,
        log_trainable_params: bool = True,
        log_non_trainable_params: bool = True,

    ):
        super().__init__()
        self._log_stats = DictConfig({
            "log_total_params": log_total_params,
            "log_trainable_params": log_trainable_params,
            "log_non_trainable_params": log_non_trainable_params,
        })

    @rank_zero_only
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:

        stats = {}
        if self._log_stats.log_total_params:
            stats["model/params_total"] = sum(p.numel() for p in pl_module.parameters())
        if self._log_stats.log_trainable_params:
            stats["model/params_trainable"] = sum(p.numel() for p in pl_module.parameters()
                                                  if p.requires_grad)
        if self._log_stats.log_non_trainable_params:
            stats["model/params_non_trainable"] = sum(p.numel() for p in pl_module.parameters()
                                                      if not p.requires_grad)
        if trainer.loggers is not None:
            for logger in trainer.loggers:
                logger.log_hyperparams(stats)
