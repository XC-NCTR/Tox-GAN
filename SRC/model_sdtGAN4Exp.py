import argparse
import os
import numpy as np
import math

from torch import autograd
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20000, help="number of epochs of training")
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
parser.add_argument("--interval", type=int, default=500, help="interval")
# parser.add_argument("--model", type=str, help="dose_time")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False
# cuda = False
device = torch.device("cuda" if cuda else "cpu")
torch.manual_seed(0)


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
            *block(4096, 2048),
            *block(2048, 2048),
            nn.Linear(2048, opt.Exp_dim),
            nn.Tanh()
        )

    def forward(self, noise, Stru, Time, Dose):
        # Concatenate condition and noise to produce input
        gen_input = torch.cat([noise, Stru, Time, Dose], -1)
        Exp = self.model(gen_input)
        return Exp


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.Stru_dim + opt.Dose_dim + opt.Time_dim + opt.Exp_dim, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4096, 1024),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 256),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, Exp, Stru, Time, Dose):
        # Concatenate condition and real_Exp to produce input
        d_in = torch.cat((Exp, Stru, Time, Dose), -1)
        validity = self.model(d_in)
        return validity


# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    # adversarial_loss.cuda()

# Configure data loader
path = r'../' # please set your own path
Stru = pd.read_csv(os.path.join(path, 'Data', 'MolDescriptors.tsv'), index_col=0, sep="\t")
Exp = pd.read_csv(os.path.join(path, 'Data', 'ExpCode_repeat.tsv'), index_col=0, sep="\t")
CEL_info = pd.read_csv(os.path.join(path, 'Data', 'open_tggates_cel_file_attribute.csv'))

compounds = pd.read_csv(os.path.join(path, 'Data', 'Compounds_Repeat.txt'), sep="\t")
flag = [Stru.index[i] in compounds.Name.to_list() for i in range(len(Stru))]
Stru = Stru[flag]  # 138 in total
scaler = MinMaxScaler(feature_range=(-1, 1))
S = scaler.fit_transform(Stru)
Stru = pd.DataFrame(S, columns=Stru.columns, index=Stru.index)
Stru = Stru.iloc[0:110]  # ~80% as training set
flag = (CEL_info.ORGAN == 'Liver') & (CEL_info.SINGLE_REPEAT_TYPE == 'Repeat') & (CEL_info.DOSE_LEVEL != 'Control')
CEL_info = CEL_info[flag]
# for dose in CEL_info.DOSE_LEVEL.unique():
#     for time in CEL_info.SACRIFICE_PERIOD.unique():
#         model = f'{dose}_{time}'
#         CELs = CEL_info[(CEL_info.DOSE_LEVEL == dose) & (CEL_info.SACRIFICE_PERIOD == time)]
# CEL_info = CEL_info[(CEL_info.DOSE_LEVEL == opt.model.split('_')[0]) &
#                     (CEL_info.SACRIFICE_PERIOD == opt.model.split('_')[1]+' day')]

scaler = MinMaxScaler(feature_range=(-1, 1))
E = scaler.fit_transform(Exp)
Exp = pd.DataFrame(E, columns=Exp.columns, index=Exp.index)

S = pd.DataFrame(columns=Stru.columns)
E = pd.DataFrame(columns=Exp.columns)


def Time(SACRIFICE_PERIOD):
    switcher = {
        '4 day': 3 / 28,
        '8 day': 7 / 28,
        '15 day': 14 / 28,
        '29 day': 28 / 28
    }
    return switcher.get(SACRIFICE_PERIOD, 'error')


T = []


def Dose(DOSE_LEVEL):
    switcher = {
        'Low': 0.1,
        'Middle': 0.3,
        'High': 1
    }
    return switcher.get(DOSE_LEVEL, 'error')


D = []
for i in range(len(CEL_info)):
    if CEL_info.iloc[i].COMPOUND_NAME in Stru.index:
        S = S.append(Stru[Stru.index == CEL_info.iloc[i].COMPOUND_NAME])
        E = E.append(Exp.loc[CEL_info.iloc[i].BARCODE + '.CEL'])
        T.append(Time(CEL_info.iloc[i].SACRIFICE_PERIOD))
        D.append(Dose(CEL_info.iloc[i].DOSE_LEVEL))

S = torch.tensor(S.to_numpy(dtype=np.float64), device=device).float()
E = torch.tensor(E.to_numpy(dtype=np.float64), device=device).float()
scaler = MinMaxScaler(feature_range=(-1, 1))
T = scaler.fit_transform(np.array(T).reshape(len(T), -1))
T = torch.tensor(T, device=device)
scaler = MinMaxScaler(feature_range=(-1, 1))
D = scaler.fit_transform(np.array(D).reshape(len(D), -1))
D = torch.tensor(D, device=device)
dataset = torch.utils.data.TensorDataset(S, E, T, D)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


def compute_gradient_penalty(Dis, real_samples, fake_samples, Stru, Time, Dose):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1)).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = Dis(interpolates, Stru, Time, Dose)
    fake = torch.ones(real_samples.shape[0], 1).to(device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ----------
#  Training
# ----------

if not os.path.exists(os.path.join(path, 'model_sdtGAN4Exp')):
    os.makedirs(os.path.join(path, 'model_sdtGAN4Exp'))

for epoch in range(opt.n_epochs):
    # start = time.time()
    for i, (Stru, Exp, Time, Dose) in enumerate(dataloader):
        batch_size = Exp.shape[0]

        # Adversarial ground truths
        valid = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sampling noise and Stru, Time, Dose as generator input
        z = torch.randn(batch_size, opt.Z_dim).to(device)

        # Generate a batch of Exp
        gen_Exp = generator(z, Stru, Time, Dose)

        validity_real = discriminator(Exp, Stru, Time, Dose)
        # # Loss for real Exp

        validity_fake = discriminator(gen_Exp.detach(), Stru, Time, Dose)
        # # Loss for fake Exp

        gradient_penalty = compute_gradient_penalty(discriminator, Exp, gen_Exp, Stru, Time, Dose)
        # Adversarial loss
        d_loss = -torch.mean(validity_real) + torch.mean(validity_fake) + lambda_gp * gradient_penalty
        # # Total discriminator loss

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if (epoch * len(dataloader) + i) % opt.n_critic == 0:
            # Generate a batch of Exp
            gen_Exp = generator(z, Stru, Time, Dose)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake Exp
            validity = discriminator(gen_Exp, Stru, Time, Dose)
            g_loss = -torch.mean(validity)
            g_loss.backward()
            optimizer_G.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch + 1, opt.n_epochs, i + 1, len(dataloader), d_loss.item(), g_loss.item())
        )
    if epoch % opt.interval == 0:
        torch.save(generator.state_dict(), os.path.join(path, 'model_sdtGAN4Exp', 'generator_{}'.format(epoch)))
    # end = time.time()
    # print('time for epoch {}:'.format(epoch + 1), end - start)
