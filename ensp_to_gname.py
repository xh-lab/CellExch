import numpy as np
import pandas as pd
from tqdm import tqdm

lr_ensp = pd.read_csv('/home/jby2/XH/CellExch/dataset1/generation/confidence_lr_ensp.csv', header=None)  # LRI that predicted by model
gn_ensp = pd.read_csv("/home/jby2/XH/CellExch/mart_export.txt", delimiter='\t', header=0).fillna('')  # file of ensp to gene name 

lr_gene_name = []

# if there is some ensp not in this file then find it on internet 
for i in tqdm(range(lr_ensp.shape[0])):  
    a = lr_ensp.iloc[i][0]  # ith lri
    l = a[: 15]  # ensp of ligand
    r = a[15: ]  
    temp = []
    l_sig = 0  # if it's 0, it indicates that the gene name of this ensp is still not find
    r_sig = 0
    for j in tqdm(range(gn_ensp.shape[0])):
        stable_id = gn_ensp.iloc[j]['Protein stable ID']
        gene_name = gn_ensp.iloc[j]['Gene name']
        if l == stable_id:
            l_sig = 1
            l_name = gene_name
        if r == stable_id:
            r_sig = 1
            r_name = gene_name
        if l_sig == r_sig and l_sig == 1 and r_sig == 1:
            temp.append(l_name)
            temp.append(r_name)
            lr_gene_name.append(temp)
            break
        
            
print(lr_gene_name)
df = pd.DataFrame(lr_gene_name)
df.to_csv("/home/jby2/XH/CellExch/dataset1/generation/LRI_predicted.csv", index=False)
          