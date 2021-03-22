import numpy as np
import pytorch_lightning as pl
from torch import nn
import torch
from EcgDataModule import EcgDataModule


class EcgModel(pl.LightningModule):
    def __init__(self,
                 num_kernels=10_000,
                 input_len=140,
                 lr=1e-3):
        super().__init__()
        self._create_model(num_kernels,
                           input_len,
                           lr)

    def _create_model(self,
                      num_kernels=10_000,
                      input_len=140,
                      lr=1e-3,
                      reg_const=1e-4):
        self.input_len = input_len
        self.num_kernels = num_kernels
        self.convs = [None] * num_kernels
        self.kernel_lens_bins = np.array([7, 9, 11])
        self.kernel_sizes = np.random.choice(self.kernel_lens_bins,
                                             self.num_kernels)
        for _ in range(num_kernels):
            dilation = 2 ** np.random.uniform(0, np.log2((input_len - 1) / (self.kernel_sizes[_] - 1)))
            dilation = int(dilation)
            padding = ((self.kernel_sizes[_] - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
            self.convs[_] = nn.Conv1d(in_channels=1,
                                      out_channels=1,
                                      kernel_size=self.kernel_sizes[_],
                                      dilation=dilation,
                                      stride=1,
                                      padding=padding)

            self.convs[_].requires_grad = False
        n_classes = 5
        self._linear = nn.Linear(in_features=self.num_kernels * 2, out_features=n_classes)
        self.softmax = nn.Softmax()

        self._criterion = nn.CrossEntropyLoss(reduction="mean")#MSELoss(reduction='mean')
        self.lr = lr
        self.reg_const = reg_const

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.reg_const)
        return optimizer

    def __calc_features(self, padding, dilation, kernel_size, feature_map):
        output_size = self.input_len + 2*padding - dilation*(kernel_size - 1)
        ppv = torch.Tensor([[len(torch.nonzero(feature))] for feature in feature_map]) / output_size
        feature2 = torch.max(feature_map, dim=2)[0]
        return ppv, feature2

    def __common_step(self, batch, batch_index):
        data, labels = batch
        features_list = list()
        for i, layer in enumerate(self.convs):
            res = layer(data)
            feature = self.__calc_features(padding=layer.padding[0],
                                           dilation=layer.dilation[0],
                                           kernel_size=layer.kernel_size[0],
                                           feature_map=res)
            features_list.append(feature[0])
            features_list.append(feature[1])
        features_list = tuple(features_list)

        all_features = torch.stack(features_list, axis=1).reshape(-1, self.num_kernels*2)

        out = self._linear(all_features) # features.shape == batch,#kernels*2

        loss = self._criterion(out, labels)

        return loss

    def training_step(self, batch, batch_ind):
        return self.__common_step(batch, batch_ind)

    def validation_step(self, batch, batch_ind):
        return self.__common_step(batch, batch_ind)

    def test_step(self, batch, batch_ind):
        return self.__common_step(batch, batch_ind)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        result = EvalResult()
        accuracy = pl.metrics.Accuracy()

if __name__ == "__main__":
    data_module = EcgDataModule(auto_download=False)
    model = EcgModel()

    trainer = pl.Trainer()
    trainer.fit(model=model,
                datamodule=data_module)
