import subprocess
from pathlib import Path
import mmap
import pickle
from itertools import chain
from datasets import load_dataset
import numpy as np
from pytorch_lightning import LightningDataModule
import tiktoken

from src.datamodules.datasets.utils import get_dataset_provider
from src.datamodules.datasets.detokenize import DATASET_DETOKENIZE_REGISTRY
from src.utils.utils import get_logger


logger = get_logger(__file__)

class LMDataModule(LightningDataModule):
    """Lightning DataModule for Language Modeling task.
    """
    def __init__(
        self,
        dataset_name,
        tokenizer_name,
        dataset_config = None,
        cache_dir = None,
        create_val_split = False,
        val_ratio = 0.005,
        val_split_seed = 1000,
        add_eot = True,
        detokenize_first = False,
        num_proc = 1,
        save_to_disk = True,
        remove_bin_files = False,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.dataset_config = dataset_config
        self.cache_dir = cache_dir
        self.create_val_split = create_val_split
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.add_eot = add_eot
        self.detokenize_first = detokenize_first
        self.num_proc = num_proc
        self.save_to_disk = save_to_disk
        self.remove_bin_files =remove_bin_files

        # check tokenizer supported by OpenAI's tiktokn library
        assert self.tokenizer_name in ("r50k_base", "gpt2", "p50k_base","cl100k_base")

    @property
    def _cache_dir_identifier(self):
        """An identifier for a specific version of tokenized dataset.
        """
        return (
            f"tokenizer_name-{self.tokenizer_name}-val_ratio-{self.val_ratio}-"
            f"val_split_seed-{self.val_split_seed}-add_eot-{self.add_eot}-"
            f"detokenize-{self.detokenize_first}"
        )

    def prepare_data(self):
        """Download / Load raw dataset from disk then tokenize.
        """
        # if `cache_dir` is not provided, download dataset from HuggingFace Hub.
        if self.cache_dir is None:
            load_dataset(self.dataset_name, self.dataset_config)
        # else load dataset from cache
        else:
            self.process_dataset(self.dataset_name)


    def process_dataset(self, dataset_name):
        cache_dir = (
            None if self.cache_dir is None else self.cache_dir / self._cache_dir_identifier
        )
        if cache_dir:
            if cache_dir.is_dir():
                return self._load_from_cache_dir(cache_dir)

        # load raw dataset
        try:
            # try to load raw dataset from disk if already downloaded
            # and moved to a custom disk location (might be different than default
            # HuggingFace cache location)
            raw_datasets = get_dataset_provider(dataset_name)()
        except ValueError:
            # else load it from HuggingFace Hub or its cache location
            raw_datasets = load_dataset(dataset_name, self.dataset_config)

        # create validation split from training split if necessary
        if self.create_val_split:
            assert ("valid", "val", "validation") not in raw_datasets.keys()
            raw_datasets = raw_datasets["train"].train_test_split(
                test_size=self.val_ratio, seed=self.val_split_seed,
                shuffle=True,   # otherwise Test split will always be at the end
            )
            raw_datasets["valid"] = raw_datasets["test"]

        # TODO: detokenize wikitext-2/103 datasets first
        # wikitext datasets come in already white-space tokenized
        # Flash-attention authors suggested to detokenize first for better
        # few-shot performance because the format will be closer to OpenWebText
        if self.detokenize_first:
            if dataset_name in DATASET_DETOKENIZE_REGISTRY:
                detokenize_fn = DATASET_DETOKENIZE_REGISTRY[dataset_name]
                raw_datasets = raw_datasets.map(
                    lambda example: {"text": detokenize_fn(example["text"])},
                    batched=False,
                    num_proc=max(self.num_proc, 1),
                    desc="Running dataset detokenization..."
                )

        # use OpenAI's original tiktoken
        # r50k_base = gpt2
        # TODO: tiktoken has updated support for GPT-3.5/4 models (not the same as GPT2)
        # see:
        # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        # try those?
        tokenizer = tiktoken.get_encoding(self.tokenizer_name)

        # get column names from raw datasets
        # these will be dropped and replaced with columns e.g. "input_ids" after tokenization
        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        # optionally add EOS (end-of-sequence) token for each row of text
        # supports batched tokenization mode
        # note: in tiktoken EOS is called `eot_token` instead
        if self.add_eot:
            def add_eot(seq):
                # tiktoken's eot_token comes in int not str
                # so `seq` needs to be tokenize first (into list of ints) then concat
                return seq + [tokenizer.eot_token] if seq is not None else seq
            def add_eot_batched(seqs):
                return [add_eot(seq) for seq in seqs]
            def tokenize(examples):
                return add_eot_batched(
                    tokenizer.encode_ordinary_batch(examples[text_column_name])
                )
        else:
            def tokenize(examples):
                return tokenizer.encode_ordinary_batch(examples[text_column_name])

        dtype = np.uint16 if tokenizer.n_vocab < 2**16 else np.int32

        # tokenize function
        # tokenize in batched mode and concat all token sequences into a single array
        def tokenize_concat(examples):
            # use `chain` to return a single iterable (list)
            # use `np.fromiter` to return a np.ndarray
            ids = np.fromiter(chain(*tokenize(examples)), dtype=dtype)
            # note: need to return a list for batched mode in the `.map()` API
            # otherwise an error will be raised
            return {"ids": [ids], "len": [len(ids)]}

        # tokenize
        # `tokenized_datasets` will contain two colunms: `ids` and `len`
        # in batched mode, default batch_size = 1000
        # so `tokenized_datasets` number of rows will be reduced by `batch_size` folds
        tokenized_datasets = raw_datasets.map(
            tokenize_concat,
            batched=True,
            num_proc=max(self.num_proc, 1),
            remove_columns=column_names,
            desc="Running tokenization on dataset..."
        )

        # save tokens as one large `.bin` file for each split to disk
        # for later usage by (possibly) other projects
        if self.save_to_disk:
            concat_ids = {}
            assert cache_dir is not None
            cache_dir.mkdir(parents=True, exist_ok=True)

            # note: the correctness of this function will be tested by checking
            # if the saved .bin file contains exactly the number of tokens expected
            # so that I know offsets etc. are correct
            def write_ids_to_disk(example, filename):
                with open(filename, "r+b") as f:
                    # create a memory-mapped file enough to hold data
                    mm = mmap.mmap(f.fileno(), 0)

                    # calculate correct offset position
                    # here I use `len(example["ids"])` instead of `example["len"]`
                    # just to be extra cautious
                    array_len = len(example["ids"])
                    starting_idx = example["len_offset"] - len(example["ids"])

                    arr = np.ndarray(shape=(array_len, ), dtype=dtype, buffer=mm,
                                     offset=np.dtype(dtype).itemsize * starting_idx)
                    arr[:] = example["ids"]
                    # flush buffer to save ids to disk
                    mm.flush()

            for split, ds in tokenized_datasets.items():

                # get the correct offsets for each row
                # e.g. if `len` is: [3, 4, 3]
                # then `len_offset` is: [3, 7, 10]
                # offset for each row (the starting position) is: [0, 3, 7]
                # the length of the entire concatenated array will be len_offset[-1] = 10
                tokenized_datasets[split] = ds.add_column("len_offset", np.cumsum(ds["len"]))
                array_len = tokenized_datasets[split][-1]["len_offset"]

                # to use memory-mapped file as a buffer
                # first need to create an empty file on disk with the size
                # capable of holding all tokens
                filename = cache_dir / f"{split}.bin"
                # note: copied from Flash-Attention code base
                subprocess.run(['truncate', '-s', str(array_len * np.dtype(dtype).itemsize),
                                str(filename)], check=True)

                # concat then save
                tokenized_datasets[split].map(
                    write_ids_to_disk,
                    fn_kwargs={"filename": filename},
                    batched=False,
                    num_proc=max(self.num_proc, 1),
                    desc="Concatenating tokens then save to disk..."
                )

                # load the concatented token ids lazily
                # token ids is viewed as a single numpy array for each split
                concat_ids[split] = np.memmap(
                    filename, dtype=dtype, mode="r", shape=(array_len, )
                )

        # save an extra `.npy` copy to `cache_dir` if provided
        # this is a more efficient form to work with in this project
        if cache_dir:
            self._save_to_cache_dir(concat_ids, tokenizer, cache_dir)
            # unlink the `.bin` files
            if self.remove_bin_files:
                for split in concat_ids:
                    Path(cache_dir / f"{split}.bin").unlink()

        # return token ids as a single ndarray
        return concat_ids, tokenizer


    def _save_to_cache_dir(self, ids, tokenizer, cache_dir):
        """Save token ids as .npy file and tokenizer as .pkl file to `cache_dir`
        """
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving token ids to cache at {str(cache_dir)}...")

        # save token ids
        for split, ids in ids.items():
            np.save(cache_dir / f"{split}.npy", ids)

        # TODO: save tiktoken tokenizer gives TypeError: cannot pickle 'builtins.CoreBPE' object
        try:
            with open(cache_dir / "tokenizer.pkl", "wb") as f:
                pickle.dump(tokenizer, f)
        except TypeError:
            logger.info(f"Cannot pickle tokenizer {tokenizer}. Skip saving....")
            Path(cache_dir / "tokenizer.pkl").unlink()


    def _load_from_cache_dir(self, cache_dir):
        """Load tokens from `cache_dir`.
        If using OpenAi's `tiktoken` tokenizers, load it from package.
        Else load it from saved cache.
        """
        assert cache_dir.is_dir()
        logger.info(f"Load from cache at {str(cache_dir)}...")
        try:
            concat_ids = {split: np.load(cache_dir / f"{split}.npy", mmap_mode="r")
                        for split in ["train", "valid", "test"]}
        except FileNotFoundError:
            logger.info("File not found, skipping split...")

        # do not support saving tiktoken tokenizers to cache
        # so load it from library
        try:
            with open(cache_dir / "tokenizer.pkl", "rb") as f:
                tokenizer = pickle.load(f)
        except FileNotFoundError:
            logger.info(
                f"Tokenizer is not found in cache. Loading OpenAI {self.tokenizer_name} tokenizer"
                "by default..."
            )
            tokenizer = tiktoken.get_encoding(self.tokenizer_name)

        return concat_ids, tokenizer

