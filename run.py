
from typing import Callable

import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf

# recursively look for .env files starting from the working directory
# `override=True` flag ensures new value for an existng key will be updated
dotenv.load_dotenv(override=True)

# add additional resolvers
# TODO: more resolvers?
OmegaConf.register_new_resolver("eval", eval)

# turn on TensorFloat32 cores for matmul and convolutions
import torch.backends
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# filter config
def dictconfig_filter_key(d: DictConfig, fn: Callable) -> DictConfig:
    """Only keeps keys in `d` where `fn(key)` is True. Supports nested configs.
    """
    # TODO: d.items() vs d.items_ex() ?
    return DictConfig({k: dictconfig_filter_key(v, fn) if isinstance(v, DictConfig) else v
                      for k, v in d.items_ex(resolve=False) if fn(k)})


@hydra.main(version_base="1.3", config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):

    # remove config keys that starts with `__`
    # these are internal keys used to compute other keys in the config
    config = dictconfig_filter_key(config, lambda k: not k.startswith('__'))

    # imports should be nested inside @hydra.main() to optimize tab completition
    # see https://github.com/facebookresearch/hydra/issues/934
    from src.utils import utils
    from src.train import train

    # apply several optional preprocessing steps on `config`
    # - force debugger-friendly run (no GPU / multiprocessing) if config.trainer.fast_dev_run=True
    utils.extras(config)

    # print config
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    # run mode
    mode = config.get("mode", "train")
    if mode not in ["train"]:
        raise NotImplementedError(f"mode {mode} is not supported")
    elif mode == "train":
        return train(config)



if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    # disables pylint E1120: broken due to @hydra.main()
    main()
