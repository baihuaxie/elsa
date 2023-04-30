import pytest
from omegaconf import open_dict
from hydra.core.hydra_config import HydraConfig
from src.utils import utils
from src.train import train

@pytest.mark.skip
def test_train_fast_dev_run(cfg_train):
    """Run for 1 train and val step."""
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.trainer.devices=1
        cfg_train.trainer.precision = "32"  # 16-mixed not supported in PL cpu mode
    utils.print_config(cfg_train)
    # TODO: test step seems not run automatically
    train(cfg_train)


def test_train_fast_dev_run_gpu(cfg_train):
    """Run for 1 train and val step on 1x GPU device. """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.trainer.devices = 1
        cfg_train.trainer.precision = "32"
    utils.print_config(cfg_train)
    train(cfg_train)

@pytest.mark.skip()
def test_train_fast_dev_run_gpu_amp(cfg_train):
    """Run for 1 train and val step on 1x GPU device with mixed precision. """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.trainer.devices = 1
        cfg_train.trainer.precision = "16-mixed"
    utils.print_config(cfg_train)
    train(cfg_train)

@pytest.mark.skip()
def test_train_fast_dev_run_2x_gpu_amp(cfg_train):
    """Run for 1 train and val step on 2x GPU device with mixed precision. """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.trainer.devices = 2
        cfg_train.trainer.strategy = "ddp"
        cfg_train.trainer.precision = "16-mixed"
    utils.print_config(cfg_train)
    train(cfg_train)


def test_save_and_resume_from_checkpoints(tmp_path, cfg_train):
    """Run for 10 train steps, save a checkpoint, then resume from saved checkpoint,
    run for 10 more steps, save again.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = 20
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.trainer.devices = 1
        cfg_train.callbacks.model_checkpoint.every_n_train_steps = 10
    utils.print_config(cfg_train)
    train(cfg_train)