import argparse
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from AE4Exp import build_model
from sklearn.preprocessing import MinMaxScaler
import scipy.spatial.distance as dis
import itertools
import os

#CEL_info = pd.read_csv("/account/xchen/workspace/TGx_GAN/Data/open_tggates_cel_file_attribute.csv")
#CELs = CEL_info.BARCODE.values
Exp = pd.read_csv("/account/bgong/workspace/TGP2012/Rat/in_vivo/Liver/Repeat/EXPRS.tsv.gz", sep="\t",index_col=0).T

min_max_scaler = MinMaxScaler()
E = min_max_scaler.fit_transform(Exp)
E = torch.tensor(E).float()

with open('/account/xchen/workspace/TGx_GAN/Data/hyperparameter_AE4Exp.json') as fp:
    hparam = json.load(fp)

hparam['save_path'] = '/account/xchen/workspace/TGx_GAN/AE4Exp/Exp_repeat/'

if not os.path.exists(hparam['save_path']):
    os.makedirs(hparam['save_path'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = torch.utils.data_train.TensorDataset(E)
dataloader = torch.utils.data_train.DataLoader(dataset, batch_size=256, shuffle=True)
model = build_model(hparam)
model.to(device)
model.fit(dataloader, 0)

model.load_model(hparam['save_path']+str(0)+'/model_checkpoint')
model.eval()
with torch.no_grad():
    code, _ = model(E.to(device))
    code = code.cpu().numpy()

df = pd.DataFrame(code)
df.index = Exp.index
df.to_csv('/account/xchen/workspace/TGx_GAN/Data/ExpCode_repeat.tsv', sep='\t', index=True)
