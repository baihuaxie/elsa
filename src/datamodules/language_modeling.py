

from pytorch_lightning import LightningDataModule


class LMDataModule(LightningDataModule):
    """Language Modeling datamodule.
    """
    def __init__(
        self,
        cache_dir,
    ):
        super().__init__()
        self.cache_dir = cache_dir


    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage: str) -> None:
        return super().setup(stage)

    def process_dataset(self):
        cache_dir = None if self.cache_dir is None else self.cache_dir
