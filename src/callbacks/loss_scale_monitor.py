from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy


class LossScaleMonitor(Callback):
    """Logs loss scale factor before each optimizer step.
    """

    # we only want to log loss scale before it changes
    # i.e. before optimizer step
    @rank_zero_only
    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        stats = {}

        # used with trainer.precision=16 + deepspeed w/o a precision plugin
        if isinstance(trainer.strategy, DeepSpeedStrategy):
            stats = {"scaler/scale": trainer.model.optimizer.loss_scale}

        # used if trainer.precision=16 and no strategy is set
        # in this case trainer.precision_plugin will be set to default
        # or if trainer.precision=32 + a strategy e.g. deepspeed or ddp
        # w/t a precision plugin
        if (
            hasattr(trainer, "precision_plugin") and
            hasattr(trainer.precision_plugin, "scaler")
        ):
            scaler = trainer.precision_plugin.scaler
            if scaler is not None:
                stats = {
                    "scaler/scale": scaler.get_scale(),
                    "scaler/growth": scaler._get_growth_tracker(),
                }

        if stats and trainer.loggers is not None:
            for logger in trainer.loggers:
                logger.log_metrics(stats, step=trainer.global_step)