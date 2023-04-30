"""Prepares pytest fixtures for all tests.
"""
import math
import pytest

from omegaconf import DictConfig, OmegaConf, open_dict
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
import pyrootutils

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("ceil", lambda x, y: math.ceil(x / y))

@pytest.fixture(scope="session")
def cfg_train_global() -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="config.yaml",
            return_hydra_config=True,
            overrides=["experiment=wikitext2/gpt2s"]
        )

        with open_dict(cfg):
            cfg.paths.root_dir = str(pyrootutils.find_root(indicator=".project-root"))
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_train_batches = 0.01
            cfg.trainer.limit_val_batches = 0.1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.trainer.precision = "32"
            cfg.datamodule.num_proc = 0
            cfg.datamodule.pin_memory = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope="function")
def cfg_train(cfg_train_global, tmp_path) -> DictConfig:
    cfg = cfg_train_global.copy()

    with open_dict(cfg):
        cfg.paths.log_dir = str(tmp_path)
        cfg.paths.output_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()
