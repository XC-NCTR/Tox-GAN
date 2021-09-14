import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np
from math import ceil


class build_model(nn.Module):

    def __init__(self, hyperparameter):
        super(build_model, self).__init__()
        input_size = hyperparameter['input_size']
        code_size = hyperparameter['code_size']
        lr = hyperparameter['learningRate']
        drop_rate = hyperparameter['drop_rate']
        self.epoch = hyperparameter['epoch']
        self.n_repeats = hyperparameter['n_repeats']
        self.n_splits = hyperparameter['n_splits']
        self.save_path = hyperparameter['save_path']
        self.patience = hyperparameter['patience']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = self.build_encoder(input_size, code_size, drop_rate)
        self.decoder = self.build_decoder(input_size, code_size)
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.AE_opt = torch.optim.RMSprop(params, lr=lr)
        self.AE_criterion = nn.MSELoss()

    def forward(self, x):
        x = self.encoder(x)
        x_de = self.decoder(x)
        return x, x_de

    def fit(self, dataloader, repeat):
        n = ceil(len(dataloader.dataset) / dataloader.batch_size)
        loss_per_epoch = []
        previous = -1
        for i in range(1, self.epoch + 1):
            ae, j = 0, 1
            for x in dataloader:
                x = x[0].to(self.device)
                _, o = self(x)
                self.AE_opt.zero_grad()
                AE_loss = self.AE_criterion(o, x)
                AE_loss.backward(retain_graph=True)
                self.AE_opt.step()
                tmp_loss = list(map(lambda x: round(float(x), 6), [AE_loss]))
                ae += tmp_loss[0]
                if j % 50 == 0:
                    print(
                        f'Repeat {repeat + 1}/{self.n_repeats * self.n_splits}  Epoch {i}/{self.epoch}  Iter {j}/{n} \n Loss: AE {ae / j:.6f}')
                    print()
                j += 1
            j -= 1
            print(
                f'Repeat {repeat + 1}/{self.n_repeats * self.n_splits}  Epoch {i}/{self.epoch}  Iter {j}/{n} \n Loss: AE {ae / j:.6f}')
            print()
            loss_per_epoch.append(list(map(lambda x: x / (j), [ae])))
            if len(loss_per_epoch) > self.patience:
                sum_loss = np.sum(np.array(loss_per_epoch), 1)
                current = np.argmin(sum_loss)
                if previous != current:
                    previous = current
                    self.save_model(repeat, loss_per_epoch)
                elif (previous == current) and (previous + self.patience == len(loss_per_epoch)):
                    print('===================Early Stopping===================')
                    break

    def build_encoder(self, input_size, code_size, drop_rate):
        encoder = nn.Sequential(
            nn.Linear(input_size, ceil(input_size/2)),
            nn.BatchNorm1d(ceil(input_size/2)),
            nn.Dropout(p=drop_rate, inplace=True),
            nn.ReLU(True),
            nn.Linear(ceil(input_size/2), ceil(input_size/4)),
            nn.BatchNorm1d(ceil(input_size/4)),
            nn.Dropout(p=drop_rate, inplace=True),
            nn.ReLU(True),
            nn.Linear(ceil(input_size/4), code_size),
            nn.BatchNorm1d(code_size),
            nn.Dropout(p=drop_rate, inplace=True),
            nn.ReLU(True))
        return encoder

    def build_decoder(self, input_size, code_size):
        decoder = nn.Sequential(
            nn.Linear(code_size, ceil(input_size/4)),
            nn.BatchNorm1d(ceil(input_size/4)),
            nn.ReLU(True),
            nn.Linear(ceil(input_size/4), ceil(input_size/2)),
            nn.BatchNorm1d(ceil(input_size/2)),
            nn.ReLU(True),
            nn.Linear(ceil(input_size/2), input_size),
            nn.Sigmoid())
        return decoder

    def save_model(self, repeat, loss_per_epoch):
        self.path = self.save_path + str(repeat) + '/'
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        torch.save(self.state_dict(), self.path + 'model_checkpoint')
        torch.save(loss_per_epoch, self.path + 'loss_per_epoch.pkl')

    def load_model(self, path):
        weights = torch.load(path, map_location=self.device)
        self.load_state_dict(weights)
