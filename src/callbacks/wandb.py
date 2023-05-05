from pathlib import Path
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

import wandb


def get_wandb_logger(trainer):
    """Get WandBLogger instance from Trainer."""

    if trainer.fast_dev_run:
        raise Exception(
            "WandBLogger is disabled by PL when trainer.fast_dev_run=True"
        )

    # if the first or the only logger passed to Trainer is
    # WandBLogger, return it directly
    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    # if multiple loggers are used
    if isinstance(trainer.loggers, list):
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "Using WandB related callbacks, but WandBLogger is not found in the Trainer."
    )


class WatchModel(Callback):
    """Track model parameters and gradients.

    `log` = "all", "gradients" or "parameters"
    """
    def __init__(self, log="all", log_freq=100):
        super().__init__()
        # note: don't write self.log here, self.log is a built-in method of PL
        self.log_str = log
        self.log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer)
        logger.watch(model=pl_module.model, log=self.log_str, log_freq=self.log_freq)


class UploadCodeAsArtifact(Callback):
    """Upload source code to WandB as artifact."""
    def __init__(self, code_dir: str):
        super().__init__()
        self.code_dir = code_dir

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):

        # get the current wandb run object
        logger = get_wandb_logger(trainer)
        experiment = logger.experiment

        # creat an wandb.Artifact object for codes
        code = wandb.Artifact(name="experiment-source", type="code")

        # get all .py files
        for path in Path(self.code_dir).resolve().rglob("*.py"):
            code.add_file(local_path=path, name=str(path.relative_to(self.code_dir)))

        # upload artifact
        experiment.log_artifact(code)


