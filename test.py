from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import torch
from torch_geometric.nn.conv import GATConv
from sklearn.metrics import matthews_corrcoef
from scipy.sparse import csr_matrix
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset, random_split, TensorDataset
from scipy import interp
import os
from sklearn.decomposition import PCA
# import KTBoost.KTBoost as KB
# import gpboost as gb


proteinFeature_L = pd.read_csv("/home/jby2/XH/Cell_Dialog/dataset1/ligand_res_fea.csv", header=None, index_col=None)
proteinFeature_R = pd.read_csv("/home/jby2/XH/Cell_Dialog/dataset1/receptor_res_fea.csv", header=None, index_col=None)
interaction = pd.read_csv("/home/jby2/XH/Cell_Dialog/dataset1/ligand-receptor-interaction.csv", header=None, index_col=None, skiprows=1)  

interaction = interaction.drop(columns=interaction.columns[0]).to_numpy()
proteinFeature_L = proteinFeature_L.drop(columns=proteinFeature_L.columns[0]).to_numpy()
proteinFeature_R = proteinFeature_R.drop(columns=proteinFeature_R.columns[0]).to_numpy()


L_R_fea = np.vstack((proteinFeature_L, proteinFeature_R))
L_R_fea = torch.tensor(L_R_fea, dtype=torch.float32)


whole_interaction = np.zeros((interaction.shape[0] + interaction.shape[1], interaction.shape[0] + interaction.shape[1]))

whole_interaction[:interaction.shape[0], interaction.shape[0]:] = interaction
whole_interaction[interaction.shape[0]:, :interaction.shape[0]] = interaction.T  

interaction1 = csr_matrix(whole_interaction)
row, col = interaction1.nonzero()
edge_index = np.vstack([row, col])
edge_index = torch.from_numpy(edge_index)



def Splicing_data(interaction):
    positive_feature = []
    negative_feature = []
    for i in range(np.shape(interaction)[0]):
        for j in range(np.shape(interaction)[1]):
            # temp = np.append(proteinFeature_L[i], proteinFeature_R[j])
            if int(interaction[i][j]) == 1:
                positive_feature.append((i, j))
            elif int(interaction[i][j]) == 0:
                negative_feature.append((i, j))

    negative_sample_index = np.random.choice(np.arange(len(negative_feature)), size=len(positive_feature),
                                             replace=False)
    negative_sample_feature = []
    for i in negative_sample_index:
        negative_sample_feature.append(negative_feature[i])
    label1 = np.ones((len(positive_feature), 1))
    label0 = np.zeros((len(negative_sample_feature), 1))
    label = np.vstack((label1, label0))
    sample = np.vstack((positive_feature, negative_sample_feature))
    return sample, label               

sample, label = Splicing_data(interaction)   
cancer = r'/home/jby2/XH/CellExch/dataset1'  # File directory for cancer species

# -----------------------------------------------end of data process-----------------------------------------------






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


acc_kt = 0
precision_kt = 0
recall_kt = 0
f1_kt = 0
AUC_kt = 0
AUPR_kt = 0
mcc_kt = 0
mcc_list = []
recall_list = []


pca_ = PCA(n_components=min(L_R_fea.shape[0], L_R_fea.shape[1]))
L_R_fea = pca_.fit_transform(L_R_fea)
print(L_R_fea.shape)
kf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
for fold, (train_index, test_index) in enumerate(kf.split(sample, label)):
    print(f"Fold: {fold + 1}")
    feature_train, feature_test = sample[train_index], sample[test_index]
    label_train, label_test = label[train_index], label[test_index]
    #feature_train = torch.tensor(feature_train, dtype=torch.float32)
    #feature_test = torch.tensor(feature_test, dtype=torch.float32)
    label_train = torch.tensor(label_train, dtype=torch.float32)
    label_test = torch.tensor(label_test, dtype=torch.float32)
    #print(type(label_train))
    #print(type(feature_train))
    
    model = GAT_MLP()
    # lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=9*1e-4, weight_decay=1e-7)
    criterion = torch.nn.BCELoss() 
    l_emb = []
    r_emb = []
    # 250
    for epoch in range(130):  
        model.train()
        optimizer.zero_grad()
        output = model(L_R_fea, feature_train)
        loss = criterion(output, label_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        model.eval()
        correct = 0
        with torch.no_grad():
            output = model(L_R_fea, feature_test)
            #_, predicted = torch.max(output.data, 1)
            y_ = []
            list_out = output.view(-1).tolist()
            for i in list_out:
                if i < 0.5:
                    y_.append(0)
                else:
                    y_.append(1)
            # compute mcc
            mcc_score = matthews_corrcoef(label_test.numpy(), np.array(y_).reshape(-1, 1))
            mcc_kt += mcc_score
            print(f"Matthews Correlation Coefficient: {mcc_score}")
            acc_kt += accuracy_score(label_test.numpy(), np.array(y_).reshape(-1, 1))
            print(f"acc: {accuracy_score(label_test.numpy(), np.array(y_).reshape(-1, 1))}")
            precision_kt += precision_score(label_test.numpy(), np.array(y_).reshape(-1, 1))
            print(f"precision: {precision_score(label_test.numpy(), np.array(y_).reshape(-1, 1))}")
            recall_kt += recall_score(label_test.numpy(), np.array(y_).reshape(-1, 1))
            print(f"recall: {recall_score(label_test.numpy(), np.array(y_).reshape(-1, 1))}")
            f1_kt += f1_score(label_test.numpy(), np.array(y_).reshape(-1, 1))
            print(f"f1_score: {f1_score(label_test.numpy(), np.array(y_).reshape(-1, 1))}")
            fpr_kt, tpr_kt, thresholds = roc_curve(label_test.numpy(), output.numpy())
            AUC_kt += auc(fpr_kt, tpr_kt)
            roc_auc = auc(fpr_kt, tpr_kt)
            print(f"roc_auc: {roc_auc}")
            # print(f"fpr: {fpr_kt}")
            prec_kt, rec_kt, thr = precision_recall_curve(label_test.numpy(), output.numpy())
            AUPR_kt += auc(rec_kt, prec_kt)
            aupr = auc(rec_kt, prec_kt)
            print(f"aupr: {aupr}")

    
        
        
        













