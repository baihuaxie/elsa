import os
from pathlib import Path
current_dir = Path(__file__).parent.absolute()

import dotenv
dotenv.load_dotenv(override=True)

# TODO: use project.toml for local build?
from src.datamodules.language_modeling import LMDataModule

class TestLMDataModules:
    """Testing Language Modeling data pipelines.
    """

    def test_wikitext_2(self):
        data_dir = os.getenv("DATA_DIR", current_dir.parent.parent / "data" / "nlp")
        cache_dir = data_dir / "wikitext-2" / "cache"

        datamodule = LMDataModule(
            dataset_name = "wikitext-2",
            tokenizer_name = "gpt2",
            cache_dir=cache_dir,
        )

        # wikitext-2 stats
        num_train_samples = 36718
        num_valid_samples = 3760
        num_test_samples = 4358

        datamodule.prepare_data()
