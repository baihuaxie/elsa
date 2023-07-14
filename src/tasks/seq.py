
import hydra
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MetricCollection

from src.optim.params_grouping import group_parameters_for_weight_decay
from src.utils.utils import get_logger

logger = get_logger(__file__)


class SequenceModel(LightningModule):
    """LightningModule for sequence modeling (Language Modeling, Masked Language Modeling) task.
    """
    def __init__(self, cfg, model_config=None):
        super().__init__()
        self.save_hyperparameters()

        self.config = cfg
        self.model_config = model_config if model_config is not None else cfg.model

        self.instantiate_model()
        self.instantiate_datamodule()
        self.instantiate_metrics()

    def instantiate_model(self):
        logger.info(f"Instantiating model: {str(self.model_config._target_)}")
        recursive = getattr(self.model_config, "_recursive_", False)
        self.model = hydra.utils.instantiate(self.model_config, _recursive_=recursive)

    def instantiate_datamodule(self):
        logger.info(f"Instantiating datamodule: {str(self.config.datamodule._target_)}")
        self._datamodule = hydra.utils.instantiate(self.config.datamodule)
        #self._datamodule.prepare_data()
        #self._datamodule.setup()

    def instantiate_metrics(self):
        if "eval" in self.config and "metrics" in self.config.eval:
            metrics_cfg = self.config.eval.metrics
        else:
            metrics_cfg ={"ppl": {"_target_": "src.metrics.perplexity.Perplexity"}}
        self.metrics = {metric_name: hydra.utils.instantiate(metric_cfg)
                        for metric_name, metric_cfg in metrics_cfg.items()}
        # wrap with `MetricCollection` to ensure metrics are computed correctly
        # on the same device with inputs -> see torchmetrics docs
        self.metrics = MetricCollection(self.metrics)
        self.train_metrics = self.metrics.clone(prefix="train/")
        self.val_metrics = self.metrics.clone(prefix="val/")
        self.test_metrics = self.metrics.clone(prefix="test/")

    def configure_optimizers(self):

        if "optimizer_params_grouping" in self.config.train:
            params_group = group_parameters_for_weight_decay(
                self.model,
                self.config.train.optimizer,
                **self.config.train.optimizer_params_grouping,
            )
        else:
            # params_group = self.model.parameters()
            # TODO: what is self.parameters() in PL?
            # https://github.com/HazyResearch/flash-attention/blob/main/training/src/tasks/seq.py
            params_group = self.parameters()
        optimizer = hydra.utils.instantiate(self.config.train.optimizer, params_group)
        logger.info(f"Optimizer: {str(self.config.train.optimizer._target_)}")

        # log optimizer info
        for i, g in enumerate(optimizer.param_groups):
            ntensors = len(g["params"])
            nparams = sum(p.numel() for p in g["params"])
            hparams = {k: v for k, v in g.items() if k != "params"}
            logger.info(
                f"Optimizer param group {i}: {ntensors} ntensors, "
                f"{nparams/2**20:.2f}M parameters, {hparams}"
            )

        # instantiate scheduler
        if "scheduler" not in self.config.train:
            logger.info("Scheduler not provided.")
            return optimizer
        else:
            lr_scheduler = hydra.utils.instantiate(self.config.train.scheduler, optimizer)
            logger.info(f"Learning rate scheduler: {str(self.config.train.scheduler._target_)}")
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "monitor": self.config.train.get("scheduler_monitor", "val/loss"),
                    "interval": self.config.train.get("scheduler_interval", "step")
                }
            }

    def forward(self, batch):
        return self.model(**batch)

    def step(self, batch, batch_idx):
        # my model computes loss internally so `labels` are passed too
        try:
            outputs = self.forward(batch)
        except RuntimeError as e:
            print(f"Error occurred in batch id: {batch_idx}")
            print(f"Batch contents: {batch}")
            torch.save(batch, f"error_batch_{batch_idx}.pt")
            raise e

        labels = batch["labels"] if isinstance(batch, dict) else batch[1]
        if isinstance(outputs, dict):
            return outputs["loss"], outputs["logits"], labels
        else:
            return outputs[0], outputs[1], labels   # loss, logits, labels

    def shared_step(self, batch, batch_idx, phase="train"):
        loss, logits, labels = self.step(batch, batch_idx)
        log_on_step = (
            ("eval" in self.config and self.config.eval.get("log_on_step", False))
            or phase == "train"
        )
        # detach `loss` tensor to avoid OOM
        self.log(f"{phase}/loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True)
        metrics = getattr(self, f"{phase}_metrics")
        metrics(logits, labels, loss.double())
        #import wandb
        #wandb.log({"train/loss": loss.item()})
        self.log_dict(metrics, on_step=log_on_step, on_epoch=False, prog_bar=True,
                      sync_dist=True, logger=True)
        return {"loss": loss, "logits": logits, "labels": labels}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, phase="train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, phase="val")

    def test_step(self, batch, batch_idx):
        return self.shared_sstep(batch, batch_idx, phase="test")


