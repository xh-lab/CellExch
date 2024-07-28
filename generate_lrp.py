import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from torch_geometric.nn.conv import GATConv
from tqdm import tqdm
from sklearn.decomposition import PCA


interaction_ = pd.read_csv("/home/jby2/XH/CellExch/dataset1/ligand-receptor-interaction.csv", header=None, index_col=None).to_numpy()
proteinFeature_L = pd.read_csv("/home/jby2/XH/CellExch/dataset1/ligand_res_fea.csv", header=None, index_col=None)
proteinFeature_R = pd.read_csv("/home/jby2/XH/CellExch/dataset1/receptor_res_fea.csv", header=None, index_col=None)

proteinFeature_L = proteinFeature_L.drop(columns=proteinFeature_L.columns[0]).to_numpy()
proteinFeature_R = proteinFeature_R.drop(columns=proteinFeature_R.columns[0]).to_numpy()
L_R_fea = np.vstack((proteinFeature_L, proteinFeature_R))
L_R_fea = torch.tensor(L_R_fea, dtype=torch.float32)

interaction = pd.read_csv("/home/jby2/XH/CellExch/dataset1/ligand-receptor-interaction.csv", header=None, index_col=None, skiprows=1)  
interaction = interaction.drop(columns=interaction.columns[0]).to_numpy()
whole_interaction = np.zeros((interaction.shape[0] + interaction.shape[1], interaction.shape[0] + interaction.shape[1]))
whole_interaction[:interaction.shape[0], interaction.shape[0]:] = interaction
whole_interaction[interaction.shape[0]:, :interaction.shape[0]] = interaction.T

interaction1 = csr_matrix(whole_interaction)
row, col = interaction1.nonzero()
edge_index = np.vstack([row, col])
edge_index = torch.from_numpy(edge_index)



# acquire receptor sequence
receptor_seq = np.delete(interaction_[0], 0)
receptor_seq = receptor_seq.reshape(-1, 1)
# acquire ligand sequence
temp = interaction_[1:]
ligand_seq = temp[: ,[0]]

print(receptor_seq.shape)
print(ligand_seq.shape)



def generate_sample(interaction):
    sample = []
    for i in range(np.shape(interaction)[0]):
        for j in range(np.shape(interaction)[1]):
            if int(interaction[i][j]) == 0:
                sample.append((i, j))
    return sample
    
sample = generate_sample(interaction)
#print(type(feature))
#print(type(position))
#print(np.array(feature).shape)
#print(np.array(position).shape)


# define similar model as the CellExch.py
class GAT_MLP(torch.nn.Module):
    def __init__(self):
        super(GAT_MLP, self).__init__()
        self.gat1 = GATConv(min(L_R_fea.shape[0], L_R_fea.shape[1]), min(L_R_fea.shape[0], L_R_fea.shape[1]), heads=3, concat=False)
        self.gat2 = GATConv(min(L_R_fea.shape[0], L_R_fea.shape[1]), min(L_R_fea.shape[0], L_R_fea.shape[1]), heads=3, concat=False)
        self.fc1 = nn.Linear(2 * min(L_R_fea.shape[0], L_R_fea.shape[1]), 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, feature, sample_train):    
        x = torch.tensor(feature, dtype=torch.float32)
        
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = self.gat2(x, edge_index)
        x = F.relu(x)
        
        feature_train = []
        for i in sample_train:
            # relative position of ligand and receptor
            l_temp = x[i[0]]
            r_temp = x[proteinFeature_L.shape[0]+i[1]]
            temp = np.append(l_temp.detach().numpy(), r_temp.detach().numpy())
            feature_train.append(temp)   
        
        x = torch.tensor(feature_train, dtype=torch.float32)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.sigmoid(x)
        return x
        

model = GAT_MLP()
model.load_state_dict(torch.load("/home/jby2/XH/CellExch/dataset1/final_model.pth"))
pca_ = PCA(n_components=min(L_R_fea.shape[0], L_R_fea.shape[1]))
L_R_fea = pca_.fit_transform(L_R_fea)

model.eval()
num = 0 
with torch.no_grad(): 
# predicting
    outputs = model(L_R_fea, sample)
    for i, j in tqdm(enumerate(outputs), total=outputs.shape[0]):
        if j >= 0.995:
            with open('/home/jby2/XH/CellExch/dataset1/generation/confidence_lr_ensp.csv', 'a') as file:
                file.write(f"{ligand_seq[sample[i][0]][0]}{receptor_seq[sample[i][1]][0]}\n")  # acquire LRI
            num += 1

print(num)