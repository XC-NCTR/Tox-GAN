import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem

if __name__ == '__main__':
    f = open("../Data/SMILES.tsv", 'r')
    f.readline()
    contents = f.readlines()
    f.close()

    smiles_list = []
    compound_list = []

    for content in contents:
        if len(content.split("\t")[0]) == 0:
            continue
        compound = content.split("\t")[1]
        compound_list.append(compound)
        smi = content.split("\t")[2]
        smiles_list.append(smi)

    # create descriptor calculator with all descriptors
    calc = Calculator(descriptors)
    # calculate for each molecule
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    result = calc.map(mols)
    result = list(result)
    data = [result[i].fill_missing(0).asdict() for i in range(len(result))]
    df = pd.DataFrame(data, index=compound_list)
    df.to_csv('../Data/MolDescriptors.tsv', sep='\t')
