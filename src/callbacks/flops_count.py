from typing import List

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

from src.utils.flops import has_deepspeed_profiler
from src.utils.flops import profile_deepspeed, profile_torch, flops_estimate


class FlopsCounter(Callback):
    """Count model FLOPs.
    """
    def __init__(self, profilers: List[str] = None):
        super().__init__()
        if profilers is None:
            profilers = ["estimate", "deepspeed"]
        if "deepspeed" in profilers and not has_deepspeed_profiler:
            raise ImportError("Deepspeed is not installed!")

        self.profilers = profilers

    @rank_zero_only
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        model, device = pl_module.model, pl_module.device

        if "deepspeed" in self.profilers:
            flops, macs, params = profile_deepspeed(model.to(device))
            if trainer.loggers is not None:
                for logger in trainer.loggers:
                    logger.log_hyperparams({
                        "GMACs/deepspeed": macs*1e-9,
                        "Gflops/deepspeed": flops*1e-9,
                        "BParams/deepspeed": params*1e-9,
                    })

        if "torch" in self.profilers:
            total_flops = profile_torch(model.to(device))
            if trainer.loggers is not None:
                for logger in trainer.loggers:
                    logger.log_hyperparams({
                        "Gflops/torch": total_flops*1e-9
                    })
            print(f"Total flops measured by PyTorch Profiler: {total_flops*1e-9} G")

        if "estimate" in self.profilers:
            flops_est, fwd_flops_est, bwd_flops_est = flops_estimate(model)
            if trainer.loggers is not None:
                for logger in trainer.loggers:
                    logger.log_hyperparams({
                        "Gflops/estimate": flops_est*1e-9,
                    })
            print(f"Total Fwd flops theoretical estimate: {fwd_flops_est*1e-9} G")
