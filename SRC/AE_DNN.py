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
        output_size = hyperparameter['output_size']
        AE_size = hyperparameter['AE_size']
        AE_lr = hyperparameter['AE_lr']
        DNN_lr = hyperparameter['DNN_lr']
        drop_rate = hyperparameter['drop_rate']
        self.epoch = hyperparameter['epoch']
        self.n_repeats = hyperparameter['n_repeats']
        self.n_splits = hyperparameter['n_splits']
        self.save_path = hyperparameter['save_path']
        self.patience = hyperparameter['patience']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = self.build_encoder(input_size, AE_size, drop_rate)
        self.decoder = self.build_decoder(input_size, AE_size)
        AE_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.AE_opt = torch.optim.RMSprop(AE_params, lr=AE_lr)
        self.AE_criterion = nn.MSELoss()
        self.DNN = self.build_DNN(AE_size, output_size, drop_rate)
        DNN_params = list(self.encoder.parameters()) + list(self.DNN.parameters())
        self.DNN_opt = torch.optim.Adam(DNN_params, lr=DNN_lr)
        self.DNN_criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = self.encoder(x)
        x_de = self.decoder(x)
        pred = self.DNN(x)
        return x_de, pred

    def fit(self, dataloader, repeat):
        n = ceil(len(dataloader.dataset) / dataloader.batch_size)
        loss_per_epoch = []
        previous = -1
        for i in range(1, self.epoch + 1):
            dnn, ae, j = 0, 0, 1
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                o, pred = self(x)
                self.AE_opt.zero_grad()
                AE_loss = self.AE_criterion(o, x)
                AE_loss.backward(retain_graph=True)
                self.AE_opt.step()
                self.DNN_opt.zero_grad()
                DNN_loss = self.DNN_criterion(pred, y)
                DNN_loss.backward()
                self.DNN_opt.step()
                tmp_loss = list(map(lambda x: round(float(x), 6), [DNN_loss, AE_loss]))
                dnn += tmp_loss[0]
                ae += tmp_loss[1]
                if j % 50 == 0:
                    print(
                        f'Repeat {repeat + 1}/{self.n_repeats * self.n_splits}  Epoch {i}/{self.epoch}  Iter {j}/{n} \n Loss:  DNN {dnn / j:.6f}  AE {ae / j:.6f}')
                    print()
                j += 1
            j -= 1
            print(
                f'Repeat {repeat + 1}/{self.n_repeats * self.n_splits}  Epoch {i}/{self.epoch}  Iter {j}/{n} \n Loss:  DNN {dnn / j:.6f}  AE {ae / j:.6f}')
            print()
            loss_per_epoch.append(list(map(lambda x: x / (j), [dnn, ae])))
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
            nn.Linear(input_size, 1000),
            nn.BatchNorm1d(1000),
            nn.Dropout(p=drop_rate, inplace=True),
            nn.ReLU(True),
            nn.Linear(1000, code_size),
            nn.BatchNorm1d(code_size),
            nn.Dropout(p=drop_rate, inplace=True),
            nn.ReLU(True))
        return encoder

    def build_decoder(self, input_size, code_size):
        decoder = nn.Sequential(
            nn.Linear(code_size, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(True),
            nn.Linear(1000, input_size),
            nn.Sigmoid())
        return decoder

    def build_DNN(self, input_size, output_size, drop_rate):
        DNN = nn.Sequential(
            nn.Linear(input_size, 1000),
            nn.BatchNorm1d(1000),
            nn.Dropout(p=drop_rate, inplace=True),
            nn.ReLU(True),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.Dropout(p=drop_rate, inplace=True),
            nn.ReLU(True),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.Dropout(p=drop_rate, inplace=True),
            nn.ReLU(True),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.Dropout(p=drop_rate, inplace=True),
            nn.ReLU(True),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.Dropout(p=drop_rate, inplace=True),
            nn.ReLU(True),
            nn.Linear(1000, output_size))
        return DNN

    def save_model(self, repeat, loss_per_epoch):
        self.path = self.save_path + str(repeat) + '/'
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        torch.save(self.state_dict(), self.path + 'model_checkpoint')
        pd.to_pickle(loss_per_epoch, self.path + 'loss_per_epoch.pkl')

    def load_model(self, path):
        weights = torch.load(path, map_location=self.device)
        self.load_state_dict(weights)
