from sklearn.metrics import auc, roc_curve, precision_recall_curve, accuracy_score, recall_score, f1_score, \
    precision_score
import numpy as np
import pandas as pd
from multi_adaboost_CNN import AdaBoostClassifier as Ada_CNN
import test2_CNN
import lightgbm as lgb
from tqdm import tqdm 
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



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


proteinFeature_L = pd.read_csv("/home/jby2/XH/CellExch/dataset1/ligand_res_fea.csv", header=None, index_col=None)
proteinFeature_R = pd.read_csv("/home/jby2/XH/CellExch/dataset1/receptor_res_fea.csv", header=None, index_col=None)
interaction = pd.read_csv("/home/jby2/XH/CellExch/dataset1/ligand-receptor-interaction.csv", header=None, index_col=None, skiprows=1)
interaction = interaction.drop(columns=interaction.columns[0]).to_numpy()
proteinFeature_L = proteinFeature_L.drop(columns=proteinFeature_L.columns[0]).to_numpy()
proteinFeature_R = proteinFeature_R.drop(columns=proteinFeature_R.columns[0]).to_numpy()

print(proteinFeature_L.shape)

sample = Splicing_data(proteinFeature_L, proteinFeature_R, interaction)

row=sample.shape[0]
column=sample.shape[1]
index = [i for i in range(row)]
np.random.shuffle(index)
index=np.array(index)
data_= sample[index,:]
X = data_[:, 0: column-1]
y = data_[:, column-1]

scaler = StandardScaler()
X = scaler.fit_transform(X)
pca = PCA(n_components=0.95)
pca.fit(X)
X = pca.transform(X)
print(X.shape)



acc = 0
precision = 0
recall = 0
f1 = 0
AUC = 0
AUPR = 0
fold =0

kf = StratifiedKFold(n_splits=5)
for train, test in tqdm(kf.split(X,y), total=kf.get_n_splits(X)): 
    fold = fold + 1
    feature_train, feature_test = X[train], X[test]
    target_train, target_test = y[train], y[test]
    feature_train_r = test2_CNN.reshape_for_CNN(feature_train)
    feature_test_r = test2_CNN.reshape_for_CNN(feature_test)
    print("-----------------------------------------------------")
    bdt_real_test_CNN = Ada_CNN(base_estimator=test2_CNN.baseline_model(n_features=X.shape[1]), n_estimators=10,  #n_features=5626
                                        learning_rate=1, epochs=1)
    bdt_real_test_CNN.fit(feature_train_r, target_train, 10)
    cpre_label = bdt_real_test_CNN.predict(feature_test_r)
    cscore = bdt_real_test_CNN.predict_proba(feature_test_r)
    cscore = cscore[:, 1]
    model = lgb.LGBMClassifier(learning_rate=0.1, n_estimators=1000)
    model.fit(feature_train, target_train)
    lscore = model.predict_proba(feature_test)
    lscore = lscore[:, 1]
    score = 0.4 * cscore + 0.6 * lscore
    pre_label = np.zeros(score.shape)
    for m in range(len(score)):
        if score[m] >= 0.5:
            pre_label[m] = 1
        else:
            pre_label[m] = 0
    
    acc += accuracy_score(target_test, pre_label)
    precision += precision_score(target_test, pre_label)
    recall += recall_score(target_test, pre_label)
    f1 += f1_score(target_test, pre_label)

    fpr, tpr, thresholds = roc_curve(target_test, score)
    prec, rec, thr = precision_recall_curve(target_test, score)
    AUC += auc(fpr, tpr)
    AUPR += auc(rec, prec)
    
    data = {'FPR': fpr, 'TPR': tpr}
    df = pd.DataFrame(data)
    directory = "/home/jby2/XH/CellExch/comparative_method_cellenboost/roc/dataset1/"
    df.to_csv(directory + f"{fold}" + ".csv", index=False)
    
    data = {'Recall': rec, 'Precision': prec}
    df = pd.DataFrame(data)
    directory = "/home/jby2/XH/CellExch/comparative_method_cellenboost/prc/dataset1/"
    df.to_csv(directory + f"{fold}" + ".csv", index=False)
    
acc = acc / 5
precision = precision / 5
recall = recall / 5
f1 = f1 / 5
AUC = AUC / 5
AUPR = AUPR / 5
with open("/home/jby2/XH/CellExch/comparative_method_cellenboost/AUC_AUPR/dataset1/mean_value_of_five-fold_cross-validation.txt", mode="a") as f:
    f.write("precision:{}\n".format(precision))
    f.write("recall:{}\n".format(recall))
    f.write("acc:{}\n".format(acc))
    f.write("f1:{}\n".format(f1))
    f.write("AUC:{}\n".format(AUC))
    f.write("AUPR:{}\n\n".format(AUPR))
    


