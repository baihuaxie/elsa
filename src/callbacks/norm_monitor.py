from typing import Any
from collections import OrderedDict
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy
import torch


class NormMonitor(Callback):
    """Log the scale (in L1 norm) of parameters and gradients at each
    global step.
    """

    @rank_zero_only
    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        model = pl_module.model
        named_parameters = dict(model.named_parameters())

        # [TD: 05-05-2023] verify that deepspeed scales gradients after this step
        # while AMP has scaled gradients by now
        # unscale gradients if using deepspeed
        if isinstance(trainer.strategy, DeepSpeedStrategy):
            loss_scale = trainer.model.optimizer.loss_scale
        else:
            loss_scale = 1.0

        stats = {}
        p_l1_norm, g_l1_norm = [], []
        for pn, p in named_parameters.items():
            p_abs = p.abs()
            p_abs_mean = p_abs.mean(dtype=torch.float32)
            stats[f"stats/{pn}_max"] = p_abs.max()
            stats[f"stats/{pn}_mean"] = p_abs_mean
            p_l1_norm.append(p_abs_mean * p.numel())    # sum of abs values in p
            if p.grad is not None:
                g_abs = p.grad.abs()
                g_abs_mean = g_abs.mean(dtype=torch.float32) / loss_scale
                stats[f"stats/{pn}_grad_max"] = g_abs.max() / loss_scale
                stats[f"stats/{pn}_grad_mean"] = g_abs_mean
                g_l1_norm.append(g_abs_mean * p.grad.numel())

        stats["stats/total_param_l1_norm"] = torch.stack(p_l1_norm).sum()
        if g_l1_norm:
            stats["stats/total_param_grad_l1_norm"] = torch.stack(g_l1_norm).sum()

        stats = OrderedDict(sorted(stats.items()))
        if trainer.loggers is not None:
            for logger in trainer.loggers:
                logger.log_metrics(stats, step=trainer.global_step)



class GradientScaleCheck(Callback):
    """Checks that by `on_before_optimizer_step` whether the graients have been
    unscaled or not in different mixed-precision training configurations.
    """
    def __init__(self):
        super().__init__()
        self.grad_l1_norm_on_before_optimizer_step = None
        self.grad_l1_norm_on_after_backward = None
        self.param_name = None

    def _get_grad_norm(self, model, param_name: str = None):
        if param_name is None:
            param_name, param = next(iter(model.named_parameters()))
            # TODO: param.grad=None in deepspeed when called by both hooks.
            assert param.grad is not None
        else:
            try:
                param = dict(model.named_parameters())[param_name]
            except KeyError:
                raise f"Parameter {param_name} does not exist in the model."
        param_grad = param.grad
        param_grad_l1_norm = torch.norm(param_grad, p=1)
        return param_grad_l1_norm, param_name

    def on_after_backward(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        model = pl_module.model
        self.grad_l1_norm_on_after_backward, self.param_name = self._get_grad_norm(model)
        print(f"Grad norm after backward: {self.grad_l1_norm_on_after_backward}")

    def on_before_optimizer_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args: Any) -> None:
        model = pl_module.model
        self.grad_l1_norm_on_before_optimizer_step = self._get_grad_norm(model, self.param_name)
        print(f"Grad norm before optimizer: {self.grad_l1_norm_on_before_optimizer_step}")
        print(f"Loss scale factor: {trainer.scaler.get_scale()}")
