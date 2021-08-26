import argparse
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from AE_DNN import build_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, accuracy_score, confusion_matrix, \
    roc_auc_score, matthews_corrcoef, f1_score
import os
from ast import literal_eval


def evaluate_model(realLabel, prediction):
    ACC = accuracy_score(realLabel, prediction)
    BACC = balanced_accuracy_score(realLabel, prediction)
    macro_recall = recall_score(realLabel, prediction, average='macro')
    macro_precision = precision_score(realLabel, prediction, average='macro')
    micro_recall = recall_score(realLabel, prediction, average='micro')
    micro_precision = precision_score(realLabel, prediction, average='micro')
    MCC = matthews_corrcoef(realLabel, prediction)
    AUC = roc_auc_score(realLabel, prediction)
    TN, FP, FN, TP = confusion_matrix(realLabel, prediction).ravel()
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    NPV = TN / (TN + FN)
    F1_Score = f1_score(realLabel, prediction)
    return ACC, BACC, MCC, AUC, sensitivity, specificity, NPV, TN, FP, FN, TP, F1_Score, \
           macro_recall, macro_precision, micro_recall, micro_precision


path = r'/account/xchen/workspace/TGx_GAN'

Labels = pd.read_csv("/account/xchen/workspace/TGx_GAN/Data/Necrosis_Label.txt", sep="\t")
train_Info = Labels[Labels.Training_Test == 'Training']

Exp = pd.read_csv(os.path.join(path, 'Results', 'ExpPerTreatment.tsv'), sep='\t')
Exp[['realExp', 'genExp']] = Exp[['realExp', 'genExp']].applymap(literal_eval)
Exp = np.array(Exp.realExp.to_list())
dataInfo = pd.read_csv(os.path.join(path, 'Results', 'MeasuresPerTreatment_pearsonr.tsv'), sep='\t')
data = np.zeros(shape=(len(Labels), Exp.shape[1]))
for i in range(len(Labels)):
    flag = (dataInfo.COMPOUND_ABBREVIATION == Labels.COMPOUND_ABBREVIATION[i]) & (
            dataInfo.SACRIFICE_PERIOD == Labels.SACRIFICE_PERIOD[i]) & (dataInfo.DOSE_LEVEL == Labels.DOSE_LEVEL[i])
    data[i] = Exp[flag]

with open('/account/xchen/workspace/TGx_GAN/Data/hyperparameter_AEDNN.json') as fp:
    hparam = json.load(fp)
hparam['save_path'] = "/account/xchen/workspace/TGx_GAN/AE_DNN/real/"

# rskf = RepeatedStratifiedKFold(n_splits=hparam['n_splits'], n_repeats=hparam['n_repeats'], random_state=2021)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

min_max_scaler = MinMaxScaler()
data = min_max_scaler.fit_transform(data)
Label = Labels.Necrosis_Label.values
D = torch.tensor(data).float().to(device)
L = torch.tensor(Label.reshape(len(Label), 1)).float().to(device)

with open(hparam['save_path'] + 'Performance.tsv', mode='w') as f:
    f.write('Train_Test\tACC\tBACC\tMCC\tAUC\tSensitivity\tSpecificity\tNPV\tTN\tFP\tFN\tTP\tF1_Score\t'
            'macro_recall\tmacro_precision\tmicro_recall\tmicro_precision\n')
    dataset = torch.utils.data.TensorDataset(D[Labels.Training_Test == 'Training'],
                                             L[Labels.Training_Test == 'Training'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    model = build_model(hparam)
    model.to(device)
    model.fit(dataloader, 0)

    model.eval()
    with torch.no_grad():
        pred = model(D[Labels.Training_Test == 'Training'])
        pred = torch.sigmoid(pred) > 0.5
        pred = pred.cpu().int().numpy()
        ACC, BACC, MCC, AUC, sensitivity, specificity, NPV, TN, FP, FN, TP, F1_Score, \
        macro_recall, macro_precision, micro_recall, micro_precision = evaluate_model(
            L[Labels.Training_Test == 'Training'].cpu(), pred)
        f.write(
            f'Train\t{ACC}\t{BACC}\t{MCC}\t{AUC}\t{sensitivity}\t{specificity}\t{NPV}\t{TN}\t{FP}\t'
            f'{FN}\t{TP}\t{F1_Score}\t{macro_recall}\t{macro_precision}\t{micro_recall}\t{micro_precision}\n')

        pred = model(D[Labels.Training_Test == 'Test'])
        pred = torch.sigmoid(pred)
        np.savetxt(hparam['save_path'] + 'real_pred_p.tsv', pred.cpu(), delimiter='\t')
        pred = pred > 0.5
        pred = pred.cpu().int().numpy()
        np.savetxt(hparam['save_path']+'real_pred_bool.tsv', pred, delimiter='\t', fmt='%s')


        ACC, BACC, MCC, AUC, sensitivity, specificity, NPV, TN, FP, FN, TP, F1_Score, \
        macro_recall, macro_precision, micro_recall, micro_precision = evaluate_model(
            L[Labels.Training_Test == 'Test'].cpu(), pred)
        f.write(
            f'Test\t{ACC}\t{BACC}\t{MCC}\t{AUC}\t{sensitivity}\t{specificity}\t{NPV}\t{TN}\t{FP}\t'
            f'{FN}\t{TP}\t{F1_Score}\t{macro_recall}\t{macro_precision}\t{micro_recall}\t{micro_precision}\n')
