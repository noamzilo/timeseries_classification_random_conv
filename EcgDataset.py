from abc import ABC

import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class EcgDataset(Dataset, ABC):
    def __init__(self, df):
        super(EcgDataset, self).__init__()

        self._df = df
        self._labels_df = df.iloc[:, 0]
        self._data_df = df.iloc[:, 1:]

    @classmethod  # this is not coupled to __init__ to allow other constructors
    def from_path(cls, path: str, substring: str, extension: str):
        print(f"creating dataset from path {path}")
        assert os.path.isdir(path)
        pattern = os.path.join(path, extension)
        file_paths = glob(pattern, recursive=False)
        file_path = [s for s in file_paths if substring in s][0]

        df = pd.read_csv(file_path, sep="\t")

        return cls(df)

    def __len__(self):
        return self._data_df.shape[0]

    def __getitem__(self, index):
        item = self._data_df.iloc[index, :].values
        return item

    def plot_sample(self):
        plt.figure()
        plt.plot(self._data_df.iloc[50:70, :].values.T)
        plt.show()


if __name__ == "__main__":
    def main():
        from paths import dataset_path
        print("WARNING: download the dataset first!")
        train_dataset = EcgDataset.from_path(dataset_path, substring="TRAIN", extension=r"*.tsv")
        test_dataset = EcgDataset.from_path(dataset_path, substring="TEST", extension=r"*.tsv")
        train_dataset.plot_sample()
        test_dataset.plot_sample()

        print(f"train len: {len(train_dataset)}. test len: {len(test_dataset)}.")

        test_entry_50 = test_dataset[50]
        plt.figure()
        plt.title("test entry 50")
        plt.plot(test_entry_50)
        plt.show()
    main()