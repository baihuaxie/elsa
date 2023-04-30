import os
from pathlib import Path
current_dir = Path(__file__).parent.absolute()

import math
import dotenv
dotenv.load_dotenv(override=True)

import torch

# TODO: use project.toml for local build?
from src.datamodules.language_modeling import LMDataModule


class TestLMDataModules:
    """Testing Language Modeling data pipelines.
    """

    def test_wikitext_2(self):
        data_dir = os.getenv("DATA_DIR", current_dir.parent.parent / "data" / "nlp")
        cache_dir = data_dir / "wikitext-2" / "cache"
        max_seq_len = 1024
        batch_size = 8

        datamodule = LMDataModule(
            dataset_name = "wikitext",
            tokenizer_name = "r50k_base",
            dataset_config = 'wikitext-2-raw-v1',
            cache_dir=cache_dir,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            val_ratio=0.005,
            val_split_seed=1000,
            add_eot=False,
            detokenize_first=False,
            num_proc=1,
        )

        # wikitext-2 stats
        num_train_tokens = 2391884  # 2.4M
        num_val_tokens = 247289
        num_test_tokens = 283287

        datamodule.prepare_data()
        datamodule.setup()
        train_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()
        test_dataloader = datamodule.test_dataloader()

        # check dataloader size are correct
        # dataloader size = number of batches
        # number of batches = number of sequences / batch_size
        # note: last batch might be incomplete, so need to be rounded up using `math.ceil()`
        # number of sequences = (number of tokens-1) // max_seq_len
        # note: `drop_last` was set to True for LM datasets, so use floor div
        # note: -1 to discount labels shifted one position to the right (one less token used)
        assert len(train_dataloader) == math.ceil(
            ((num_train_tokens-1) // max_seq_len) / batch_size
        )
        assert len(val_dataloader) == math.ceil(
            ((num_val_tokens-1) // max_seq_len) / batch_size
        )
        assert len(test_dataloader) == math.ceil(
            ((num_test_tokens-1) // max_seq_len) / batch_size
        )

        # check a batch of data has the correct tensor shape
        for loader in [train_dataloader, val_dataloader, test_dataloader]:
            x, y = next(iter(loader))
            assert x.dim() == 2     # (batch, seq_len,)
            assert x.shape == (batch_size, max_seq_len)
            assert x.dtype == torch.long
            # check language modeling labels are properly shifted one position
            # to the right of input tokens
            assert torch.allclose(x[:, 1:], y[:, :-1])


    def test_wikitext_103(self):
        data_dir = os.getenv("DATA_DIR", current_dir.parent.parent / "data" / "nlp")
        cache_dir = data_dir / "wikitext-103" / "cache"
        max_seq_len = 1024
        batch_size = 8

        datamodule = LMDataModule(
            dataset_name = "wikitext",
            tokenizer_name="r50k_base",
            dataset_config = 'wikitext-103-raw-v1',
            cache_dir=cache_dir,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            val_ratio=0.005,
            val_split_seed=1000,
            add_eot=False,
            detokenize_first=False,
        )

        # wikitext-103 stats
        num_train_tokens = 117920140    # 117M
        num_val_tokens = 247289
        num_test_tokens = 283287

        datamodule.prepare_data()
        datamodule.setup()
        train_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()
        test_dataloader = datamodule.test_dataloader()

        assert len(train_dataloader) == math.ceil(
            ((num_train_tokens-1) // max_seq_len) / batch_size
        )
        assert len(val_dataloader) == math.ceil(
            ((num_val_tokens-1) // max_seq_len) / batch_size
        )
        assert len(test_dataloader) == math.ceil(
            ((num_test_tokens-1) // max_seq_len) / batch_size
        )

        # check a batch of data has the correct tensor shape
        for loader in [train_dataloader, val_dataloader, test_dataloader]:
            x, y = next(iter(loader))
            assert x.dim() == 2     # (batch, seq_len,)
            assert x.shape == (batch_size, max_seq_len)
            assert x.dtype == torch.long
            # check language modeling labels are properly shifted one position
            # to the right of input tokens
            assert torch.allclose(x[:, 1:], y[:, :-1])


    def test_openwebtext(self):
        data_dir = os.getenv("DATA_DIR", current_dir.parent.parent / "data" / "nlp")
        cache_dir = data_dir / "openwebtext" / "cache"
        max_seq_len = 1024
        batch_size = 16

        datamodule = LMDataModule(
            dataset_name = "openwebtext",
            tokenizer_name = "r50k_base",
            dataset_config=None,
            cache_dir=cache_dir,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            add_eot=True,
            detokenize_first=False,
            num_proc=4,
        )

        # wikitext-2 stats
        # obtained by print(len(concat_ids[split]))
        # note: openwebtext test split is dummy, so we don't use it
        num_train_tokens = 9035582198   # 9B
        num_val_tokens = 4434897

        datamodule.prepare_data()
        datamodule.setup()
        train_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()

        assert len(train_dataloader) == math.ceil(
            ((num_train_tokens-1) // max_seq_len) / batch_size
        )
        assert len(val_dataloader) == math.ceil(
            ((num_val_tokens-1) // max_seq_len) / batch_size
        )

        # check a batch of data has the correct tensor shape
        for loader in [train_dataloader, val_dataloader]:
            x, y = next(iter(loader))
            assert x.dim() == 2     # (batch, seq_len,)
            assert x.shape == (batch_size, max_seq_len)
            assert x.dtype == torch.long
            # check language modeling labels are properly shifted one position
            # to the right of input tokens
            assert torch.allclose(x[:, 1:], y[:, :-1])
