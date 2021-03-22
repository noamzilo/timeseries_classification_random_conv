import pytorch_lightning as pl
from torch.utils.data import DataLoader
from EcgDataset import EcgDataset
from typing import Optional
import torch
from paths import data_path, dataset_path
import matplotlib.pyplot as plt
from pathlib import Path
import os
import requests
from zipfile import ZipFile
import traceback
import sys


class EcgDataModule(pl.LightningDataModule):
    def __init__(self, auto_download=True):
        super(EcgDataModule, self).__init__()
        self._auto_download = auto_download
        self._data_path = data_path
        self._ecg_dataset_path = dataset_path

        self._train_dataset: Optional[EcgDataset] = None
        self._val_dataset: Optional[EcgDataset] = None
        self._test_dataset: Optional[EcgDataset] = None

        self._batch_size = 16
        self._num_workers = 1
        self._val_ratio = 0.1

    def prepare_data(self, *args, **kwargs):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

        # for manual download see ./zip_password.txt
        if not self._auto_download:
            return

        def download_dataset():
            Path(data_path).mkdir(parents=True, exist_ok=True)  # assuming Python >= 3.5
            url_str = 'https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip'
            zip_out_path = os.path.join(data_path, "UCRArchive_2018.zip")

            if not os.path.exists(zip_out_path):
                try:
                    print("starting to download dataset zip")
                    file_content = requests.get(url_str, timeout=10, verify=False).content
                except requests.RequestException as ex:
                    traceback.print_exc(ex)
                    sys.stderr.write(r"Auto download failed. For manual download see ./zip_password.txt")
                    raise
                print("downloaded dataset zip. Takes ~1min")
                print("writing zip.")
                with open(zip_out_path, 'wb') as out_file:
                    out_file.write(file_content)

            print("extracting zip. Takes several minutes, manual is fast.")
            with ZipFile(zip_out_path, 'r') as zip_ref:
                zip_ref.extractall(data_path, pwd=b"someone")
            print("done extracting.")

        download_dataset()

    def setup(self, stage: Optional[str] = None):
        # make assignments here (val/train/test split)
        # called on every process in DDP

        self._train_dataset = EcgDataset.from_path(self._ecg_dataset_path, substring="TRAIN", extension=r"*.tsv")
        full_test_dataset = EcgDataset.from_path(self._ecg_dataset_path, substring="TRAIN", extension=r"*.tsv")
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
        module = EcgDataModule(auto_download=False)
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
