from typing import Dict
import os
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor


class ModelCheckpointMine(pl.callbacks.model_checkpoint.ModelCheckpoint):
    """Removes any existing `last.ckpt` file before saving a new one.
    With this checkpoint, there will always be just one `last.ckpt` file
    under the checkpoint directory, which points to the lastest saved step.
    """
    # TODO: should I use rank_zero_only here?
    @rank_zero_only
    def _save_last_checkpoint(
        self,
        trainer: "pl.Trainer",
        monitor_candidates: Dict[str, Tensor]
    ) -> None:
        if not self.save_last:
            return

        filepath = self.format_checkpoint_name(
            monitor_candidates, self.CHECKPOINT_NAME_LAST
        )

        # remove existing `last.ckpt`
        # at this point the training has already resumed, so this is not needed
        if os.path.exists(filepath):
            os.remove(filepath)

        self.last_model_path = filepath
        self._save_checkpoint(trainer, filepath)

        # [06-18-2023]: the following are the original code in PL's source
        # it automatically appends a version number, so I will be saving
        # last.ckpt, last-v1.ckpt, last-v2.ckpt, etc.
        # note: it seems that self.last_model_path is not loaded in the Trainer
        # the Trainer saves this argument as a key-value pair by the
        # corresponding model checkpoint callback instance, so the line to
        # remove previous checkpoints will never be called here.

        #version_cnt = self.STARTING_VERSION
        #while self.file_exists(filepath, trainer) and filepath != self.last_model_path:
        #    filepath = self.format_checkpoint_name(
        #        monitor_candidates, self.CHECKPOINT_NAME_LAST, ver=version_cnt
        #    )
        #    version_cnt += 1
        # set the last model path before saving because it will be part of the state.
        #previous, self.last_model_path = self.last_model_path, filepath
        #self._save_checkpoint(trainer, filepath)
        #if previous and previous != filepath:
        #    self._remove_checkpoint(trainer, previous)