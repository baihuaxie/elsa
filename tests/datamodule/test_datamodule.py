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
            cache_dir=cache_dir,
        )
        datamodule.prepare_data()
