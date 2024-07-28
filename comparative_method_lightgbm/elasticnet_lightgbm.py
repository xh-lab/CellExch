import numpy as np
import pandas as pd
import scipy.io as sio
import lightgbm as lgb
from L1_Matine import elasticNet
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from tqdm import tqdm


proteinFeature_L = pd.read_csv("/home/jby2/XH/CellExch/dataset4/ligand_res_fea.csv", header=None, index_col=None)
proteinFeature_R = pd.read_csv("/home/jby2/XH/CellExch/dataset4/receptor_res_fea.csv", header=None, index_col=None)
interaction = pd.read_csv("/home/jby2/XH/CellExch/dataset4/ligand-receptor-interaction.csv", header=None, index_col=None, skiprows=1)  

interaction = interaction.drop(columns=interaction.columns[0]).to_numpy()
proteinFeature_L = proteinFeature_L.drop(columns=proteinFeature_L.columns[0]).to_numpy()
proteinFeature_R = proteinFeature_R.drop(columns=proteinFeature_R.columns[0]).to_numpy()


def Splicing_data(proteinFeature_L, proteinFeature_R, interaction):
    positive_feature = []
    negative_feature = []
    for i in range(np.shape(interaction)[0]):  
        for j in range(np.shape(interaction)[1]):  
            temp = np.append(proteinFeature_L[i], proteinFeature_R[j])
            if int(interaction[i][j]) == 1:
                temp = np.hstack((temp, np.array(1)))
                positive_feature.append(temp) 
            elif int(interaction[i][j]) == 0:
                temp = np.hstack((temp, np.array(0)))
                negative_feature.append(temp)  
                
    negative_sample_index = np.random.choice(np.arange(len(negative_feature)), size=len(positive_feature),
                                             replace=False)  
    negative_sample_feature = []
    for i in negative_sample_index:
        negative_sample_feature.append(negative_feature[i])  
    sample = np.vstack((positive_feature, negative_sample_feature))  
    #label1 = np.ones((len(positive_feature), 1))  
    #label0 = np.zeros((len(negative_sample_feature), 1))  
    #label = np.vstack((label1, label0))  
    return sample
    
sample = Splicing_data(proteinFeature_L, proteinFeature_R, interaction)
row=sample.shape[0]
column=sample.shape[1]
index = [i for i in range(row)]
np.random.shuffle(index)
index=np.array(index)
data_= sample[index,:]
shu = data_[:, 0: column-1]
label=data_[:, column-1]
label[label==0]=-1
data_1,mask1=elasticNet(shu, label)#
X=data_1
label[label==-1]=0
y=label



acc = 0
precision = 0
recall = 0
f1 = 0
AUC = 0
AUPR = 0
fold = 0


skf= StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
for train, test in tqdm(skf.split(X,y), total=skf.get_n_splits(X)): 
    fold = fold + 1
    feature_train, feature_test = X[train], X[test]
    target_train, target_test = y[train], y[test]
    gbm = lgb.LGBMClassifier(n_estimators=500,max_depth=15,learning_rate=0.2)
    gbm.fit(feature_train, target_train.ravel())
    y_ = gbm.predict(feature_test)
    prob_output = gbm.predict_proba(feature_test)
    y_prob_1 = np.arange(0, dtype=float)
    for i in prob_output:
        y_prob_1 = np.append(y_prob_1, i[1])

    acc += accuracy_score(target_test, y_)
    precision += precision_score(target_test, y_)
    recall += recall_score(target_test, y_)
    f1 += f1_score(target_test, y_)

    fpr, tpr, thresholds = roc_curve(target_test, y_prob_1)
    prec, rec, thr = precision_recall_curve(target_test, y_prob_1)
    AUC += auc(fpr, tpr)
    AUPR += auc(rec, prec)
    
    data = {'FPR': fpr, 'TPR': tpr}
    df = pd.DataFrame(data)
    directory = '/home/jby2/XH/CellExch/comparative_method_lightgbm/roc/dataset4/'
    df.to_csv(directory + f"{fold}" + ".csv", index=False)
    
    data = {'Recall': rec, 'Precision': prec}
    df = pd.DataFrame(data)
    directory = '/home/jby2/XH/CellExch/comparative_method_lightgbm/prc/dataset4/'
    df.to_csv(directory + f"{fold}" + ".csv", index=False)
    
    
    
acc = acc / 5
precision = precision / 5
recall = recall / 5
f1 = f1 / 5
AUC = AUC / 5
AUPR = AUPR / 5
with open("/home/jby2/XH/CellExch/comparative_method_lightgbm/AUC_AUPR/dataset4/mean_value_of_five-fold_cross-validation.txt", mode="a") as f:
    f.write("precision:{}\n".format(precision))
    f.write("recall:{}\n".format(recall))
    f.write("acc:{}\n".format(acc))
    f.write("f1:{}\n".format(f1))
    f.write("AUC:{}\n".format(AUC))
    f.write("AUPR:{}\n\n".format(AUPR))

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
  