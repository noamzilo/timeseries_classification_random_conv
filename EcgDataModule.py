import pytorch_lightning as pl
from torch.utils.data import DataLoader
from EcgDataset import EcgDataset
from typing import Optional
import torch
from paths import data_path
import matplotlib.pyplot as plt


class EcgDataModule(pl.LightningDataModule):
    def __init__(self):
        super(EcgDataModule, self).__init__()
        self._data_path = data_path

        self._train_dataset: Optional[EcgDataset] = None
        self._val_dataset: Optional[EcgDataset] = None
        self._test_dataset: Optional[EcgDataset] = None

        self._batch_size = 16
        self._num_workers = 1
        self._val_ratio = 0.1

    def prepare_data(self, *args, **kwargs):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

        # download from ./zip_password.txt
        pass

    def setup(self, stage: Optional[str] = None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        self._train_dataset = EcgDataset.from_path(self._data_path, substring="TRAIN", extension=r"*.tsv")
        full_test_dataset = EcgDataset.from_path(self._data_path, substring="TRAIN", extension=r"*.tsv")
        val_size = int(len(full_test_dataset) * self._val_ratio)
        test_size = len(full_test_dataset) - val_size
        self._val_dataset, self._train_dataset = torch.utils.data.random_split(full_test_dataset, [val_size, test_size])

    def train_dataloader(self):
        loader = DataLoader(self._train_dataset, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=True, drop_last=False)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self._val_dataset, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False, drop_last=False)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self._test_dataset, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False, drop_last=False)
        return loader


if __name__ == "__main__":
    def main():
        module = EcgDataModule()
        module.prepare_data()
        module.setup()
        dataloader = module.train_dataloader()

        for batch in dataloader:
            first_item = batch[0, :]
            plt.figure()
            plt.plot(first_item)
            plt.show()
            break
            
    main()
