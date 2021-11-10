import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
import argparse
from AE4Exp import build_model
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=2e-4, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--Z_dim", type=int, default=1000, help="dimension of the latent space (noise)")
parser.add_argument("--Stru_dim", type=int, default=1826, help="dimension of molecular descriptors")
parser.add_argument("--Time_dim", type=int, default=1,
                    help="dimension of final administration time point (3,7,14,28 days)")
parser.add_argument("--Dose_dim", type=int, default=1, help="dimension of dose level (low:middle:high=1:3:10)")
parser.add_argument("--Exp_dim", type=int, default=1826, help="dimension of gene expression code")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

device = torch.device("cuda" if cuda else "cpu")

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.Z_dim + opt.Stru_dim + opt.Dose_dim + opt.Time_dim, 4096, normalize=False),
            *block(4096, 4096),
            *block(4096, 4096),
            *block(4096, 2048),
            *block(2048, 2048),
            nn.Linear(2048, opt.Exp_dim),
            nn.Tanh()
        )

    def forward(self, noise, Stru, Time, Dose):
        # Concatenate Stru and noise_Exp to produce input
        gen_input = torch.cat([noise, Stru, Time, Dose], -1)
        Exp = self.model(gen_input)
        return Exp


path = r'../' # please set your own path
Treatments = pd.read_csv(os.path.join(path, 'Data', 'Treatments_test.tsv'), sep="\t")
Stru = pd.read_csv(os.path.join(path, 'Data', 'MolDescriptors_test.tsv'), index_col=0, sep="\t")
Stru0 = pd.read_csv(os.path.join(path, 'Data', 'MolDescriptors.tsv'), index_col=0, sep="\t")
scaler_Stru = MinMaxScaler(feature_range=(-1, 1))
scaler_Stru.fit(Stru0)
ExpCode = pd.read_csv(os.path.join(path, 'Data', 'ExpCode_FC_repeat.tsv'), index_col=0, sep="\t")
scaler_ExpCode = MinMaxScaler(feature_range=(-1,1))
scaler_ExpCode.fit(ExpCode)

S0 = scaler_Stru.transform(Stru)

S = pd.DataFrame()
for i in range(len(Treatments)):
    S = S.append(pd.DataFrame(S0[np.where(Stru.index == Treatments.Compound[i])]))
S = torch.tensor(S.to_numpy(), device=device).float().reshape(S.shape[0], -1)
T = [float(Treatments.Time[i].strip(' day')) / 28 for i in range(len(Treatments))]
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(np.array([3 / 28, 7 / 28, 14 / 28, 28 / 28]).reshape(4, -1))
T = scaler.transform(np.array(T).reshape(len(T), -1))
T = torch.tensor(T, device=device)


def Dose(DOSE_LEVEL):
    switcher = {
        'Low': 0.1,
        'Middle': 0.3,
        'High': 1
    }
    return switcher.get(DOSE_LEVEL, 'error')


D = [Dose(Treatments.Dose_Level[i].capitalize()) for i in range(len(Treatments))]
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(np.array([0.1, 0.3, 1]).reshape(3, -1))
D = scaler.transform(np.array(D).reshape(len(D), -1))
D = torch.tensor(D, device=device)

model_path = r'../model_sdtGAN4FC' # please change it to the folder where your well-trained model save
weights = torch.load(os.path.join(model_path, 'generator_FC'))
generator = Generator()
if cuda:
    generator.cuda()
generator.load_state_dict(weights)

generator.eval()

torch.manual_seed(0)
genExpCode = torch.tensor(()).to(device)
with torch.no_grad():
    for i in range(100):
        z = torch.randn(S.shape[0], opt.Z_dim).to(device)
        genExpCode = torch.cat((genExpCode, generator(z, S, T, D)))

genExpCode = scaler_ExpCode.inverse_transform(genExpCode.cpu().detach().numpy())

with open('../Data/hyperparameter_AE4Exp.json') as fp:
    hparam = json.load(fp)
hparam['save_path'] = '../AE4Exp/FC_repeat/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(hparam)
model.to(device)
model.load_model(hparam['save_path']+str(0)+'/model_checkpoint')
model.eval()
with torch.no_grad():
    de = model.decoder(torch.tensor(genExpCode).float().to(device))
    de = de.cpu().numpy()

Exp = pd.read_csv("../Data/logFC.tsv", sep="\t", index_col=0).T
scaler = MinMaxScaler()
scaler.fit_transform(Exp)
df = pd.DataFrame(scaler.inverse_transform(de))
#df.index = Stru.index
#df.to_csv(os.path.join(path, 'genDM_FC.tsv'), sep='\t')
np.save(os.path.join(path, 'Results', 'gen_FC_100.npy'), scaler.inverse_transform(de))