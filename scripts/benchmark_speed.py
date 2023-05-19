from typing import Dict, Callable
from pathlib import Path
from datetime import datetime
import dotenv
import hydra
from omegaconf import DictConfig, open_dict, OmegaConf
import torch

from src.utils.benchmark import benchmark_all
from src.utils.pandas_utils import reshape_nested_dict
from src.utils.seaborn_utils import create_lineplot_for_runtime

dotenv.load_dotenv(override=True)

@hydra.main(version_base="1.3", config_path="../configs/benchmark/", config_name="base.yaml")
def main(config: DictConfig):

    # set config so the fields could be modified
    OmegaConf.set_struct(config, False)

    # sequence lengths
    seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    # run benchmark on each sequence length
    results = {}
    for seq_len in seq_lens:

        # update `seq_len` in config
        with open_dict(config):
            config.config.seq_len = seq_len
        print(f"---- Benchmark Runtime vs. Sequence Length: {str(config.config.seq_len)} ----")

        # instantiate methods to benchmark
        methods : Dict[Callable] = {}
        if "baselines" in config and config.run_baselines:
            for k, method_cfg in config.baselines.items():
                if method_cfg is not None and "_target_" in method_cfg:
                    print(f"Instantiating method: <{str(method_cfg._target_)}>")
                    methods[k] = hydra.utils.instantiate(method_cfg)
        if "methods" in config:
            for k, method_cfg in config.methods.items():
                if method_cfg is not None and "_target_" in method_cfg:
                    print(f"Instantiating method: <{str(method_cfg._target_)}>")
                    methods[k] = hydra.utils.instantiate(method_cfg)

        # dummy input: (batch_size, seq_len, n_embeds)
        # TODO: batch_size = ??
        inputs = torch.randn(config.config.batch_size, config.config.seq_len,
                             config.config.n_embed, device="cuda")

        # run benchmarks
        len_results = {}
        for k, method in methods.items():
            method.to("cuda")
            fwd, bwd, cmb = benchmark_all(
                # TODO: set repeats=10
                method, inputs, repeats=10, desc=k, verbose=True, enable_amp=config.amp
            )
            len_results[k] = {"Fwd": fwd.mean, "Bwd": bwd.mean, "Cmb": cmb.mean}

        # gather results
        results[seq_len] = len_results

    # set up identifier for current run
    output_dir = Path(config.output_dir) / "speed"
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_identifier = f"n_embed-{config.config.n_embed}-amp-{config.amp}"
    if config.test_run:
        run_name = output_dir / "test"
    else:
        run_identifier = datetime.now().strftime("%m%d%Y-%H%M")
        run_name = output_dir / "-".join([benchmark_identifier, run_identifier])

    # set up output .csv file path
    if config.save_csv:
        csv_file = f"{run_name}.csv"
    else:
        csv_file = None

    # save results as pandas dataframe (long format)
    # pivot dataframe to wide format for plotting
    index = ["seq_len", "methods", "pass"]
    columns = ["pass", "methods"]
    rows = ["seq_len"]
    df = reshape_nested_dict(
        results, index, columns, rows, csv_file=csv_file,
    )

    # creat and save lineplots
    if config.save_plot:
        save_plot = f"{run_name}-Combined.png"
        print(f"Saving line plot to: {save_plot}")
        create_lineplot_for_runtime(df, pass_col="Cmb", save_plot=save_plot)



if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
