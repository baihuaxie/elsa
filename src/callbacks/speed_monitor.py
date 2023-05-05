
import time
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from omegaconf import DictConfig


class SpeedMonitor(Callback):
    """Logs time to process each training batch and time in between batches.
    """
    def __init__(
        self,
        log_intra_step_time: bool = True,
        log_inter_step_time: bool = True,
        log_training_time: bool = True,
    ):
        super().__init__()
        self._log_stats = DictConfig({
            "log_intra_step_time": log_intra_step_time,
            "log_inter_step_time": log_inter_step_time,
            "log_training_time": log_training_time,
        })

    def on_train_start(self, trainer, pl_module):
        self._stamp_epoch_start = None

    def on_train_epoch_start(self, trainer, pl_module):
        self._stamp_batch_start = None
        self._stamp_batch_end = None
        self._stamp_epoch_start = time.time()

    # skip validation time
    def on_validation_epoch_start(self, trainer, pl_module):
        self._stamp_batch_end = None

    def on_test_epoch_start(self, trainer, pl_module):
        self._stamp_batch_end = None

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):

        # stamp training batch start time
        if self._log_stats.log_intra_step_time:
            self._stamp_batch_start = time.time()

        # log inter step time
        # for the first batch, `self.stamp_batch_end` = None
        stats = {}
        if self._log_stats.log_inter_step_time and self._stamp_batch_end:
            stats["time/inter_step (ms)"] = (
                time.time() - self._stamp_batch_end
            ) * 1000

        if trainer.loggers is not None:
            for logger in trainer.loggers:
                logger.log_metrics(stats, step=trainer.global_step)

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):

        # stamp training batch end time
        if self._log_stats.log_inter_step_time:
            self._stamp_batch_end = time.time()

        # log intra step time
        stats = {}
        if self._log_stats.log_intra_step_time and self._stamp_batch_start:
            stats["time/intra_step (ms)"] = (
                time.time() - self._stamp_batch_start
            ) * 1000

        if trainer.loggers is not None:
            for logger in trainer.loggers:
                logger.log_metrics(stats, step=trainer.global_step)

    # TODO: hook when trainer.max_steps reached?
    # just want to log the time to process batches, ignore logistics
    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        stats = {}
        if self._log_stats.log_training_time and self._stamp_epoch_start:
            stats["time/training_time (ms)"] = (
                time.time() - self._stamp_epoch_start
            ) * 1000
        if trainer.loggers is not None:
            for logger in trainer.loggers:
                logger.log_metrics(stats, step=trainer.global_step)

        
