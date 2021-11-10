import pubchempy as pcp

with open('../Data/SMILES.tsv', mode='w') as f1:
    f1.write('CID\tCompoundName\tSMILES\n')
    with open('../Data/Compounds_Repeat.txt', mode='r') as f2:
        contents = f2.readlines()
        for content in contents:
            content = content.strip('\n').split('\t')
            compound = pcp.get_compounds(content[0], 'name')
            if len(compound)==0:
                CID = ''
                SMILES = ''
            else:
                compound = compound[0]
                CID = str(compound.cid)
                SMILES = compound.canonical_smiles
            f1.write(CID + "\t" + content[0] + "\t" + SMILES + "\n")
