# Copied and adapted from:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/utils/utils.py
from typing import Sequence, Optional
from pathlib import Path
import logging
import rich.tree
import rich.syntax
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning.utilities import rank_zero_only


def get_logger(name=__name__) -> logging.Logger:
    """Instantiate a multi-GPU-friendly python logger.
    Only generates logs if rank==0.
    """
    logger = logging.getLogger(name)

    # ensures all logging levels gets marked with `rank_zero_only()` decorator
    # so logs won't get multiplied for each GPU process
    # note: `getattr(logger, level)` would return a logging.Level object
    # that can be decorated by `rank_zero_only(func)`
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def extras(config: DictConfig) -> None:
    """Several optional utilities to pre-process `config`, controlled by `config` flags.
    - enable `fast_dev_run` mode that disables GPU and multiprocessing for fast debug
    """
    logger = get_logger(__name__)

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    # debuggers don't like GPUs and multiprocessing
    if config.trainer.get("fast_dev_run"):
        logger.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Optional[Sequence[str]] = (
        "paths",
        "trainer",
        "model",
        "datamodule",
        "train",
        "eval",
        "callbacks",
        "loggers",
        "seed",
        "name"
    ),
    resolve: bool = True
) -> None:
    """Pretty-print `config` as a rich Tree.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    # constructs a tree-like structure for each field and its contents in `config`
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        # if config is nested, convert to yaml format
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)
        # else converts to string
        else:
            branch_content = str(config_section)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print to console
    rich.print(tree)

    # print to log file
    # TODO: this doesn't seem to save the log file to output directory
    with open(Path(config.paths.output_dir, "config_tree.txt"), "w") as fp:
        rich.print(tree, file=fp)


