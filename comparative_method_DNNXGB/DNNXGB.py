import numpy as np
import pandas as pd
from keras.layers import Dense, Input, Dropout
from keras.layers.merging import concatenate
from keras.optimizers import SGD
from keras.models import Model
from keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef,accuracy_score, precision_score,recall_score, f1_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.manifold import TSNE
from tqdm import tqdm
from xgboost import XGBClassifier
import time

start = time.time()
def define_model():
    
    ########################################################"Channel-1" ########################################################
    
    input_1 = Input(shape=(2813, ), name='Protein_a')
    p11 = Dense(512, activation='relu', kernel_initializer='glorot_normal', name='ProA_feature_1', kernel_regularizer=l2(0.01))(input_1)
    p11 = Dropout(0.2)(p11)
    
    p12 = Dense(256, activation='relu', kernel_initializer='glorot_normal', name='ProA_feature_2', kernel_regularizer=l2(0.01))(p11)
    p12 = Dropout(0.2)(p12)
    
    p13= Dense(128, activation='relu', kernel_initializer='glorot_normal', name='ProA_feature_3', kernel_regularizer=l2(0.01))(p12)
    p13 = Dropout(0.2)(p13)
    
    p14= Dense(64, activation='relu', kernel_initializer='glorot_normal', name='ProA_feature_4', kernel_regularizer=l2(0.01))(p13)
    p14 = Dropout(0.2)(p14)
    
    ########################################################"Channel-2" ########################################################
    
    input_2 = Input(shape=(2813, ), name='Protein_b')
    p21 = Dense(512, activation='relu', kernel_initializer='glorot_normal', name='ProB_feature_1', kernel_regularizer=l2(0.01))(input_2)
    p21 = Dropout(0.2)(p21)
    
    p22 = Dense(256, activation='relu', kernel_initializer='glorot_normal', name='ProB_feature_2', kernel_regularizer=l2(0.01))(p21)
    p22 = Dropout(0.2)(p22)
    
    p23= Dense(128, activation='relu', kernel_initializer='glorot_normal', name='ProB_feature_3', kernel_regularizer=l2(0.01))(p22)
    p23 = Dropout(0.2)(p23)
    
    p24= Dense(64, activation='relu', kernel_initializer='glorot_normal', name='ProB_feature_4', kernel_regularizer=l2(0.01))(p23)
    p24 = Dropout(0.2)(p24)
   


    ##################################### Merge Abstraction features ##################################################
    
    merged = concatenate([p14,p24], name='merged_protein1_2')
    
    ##################################### Prediction Module ##########################################################
    
    pre_output = Dense(64, activation='relu', kernel_initializer='glorot_normal', name='Merged_feature_1')(merged)
    pre_output = Dense(32, activation='relu', kernel_initializer='glorot_normal', name='Merged_feature_2')(pre_output)
    pre_output = Dense(16, activation='relu', kernel_initializer='he_uniform', name='Merged_feature_3')(pre_output)


    
    pre_output=Dropout(0.2)(pre_output)

    output = Dense(1, activation='sigmoid', name='output')(pre_output)
    model = Model(inputs=[input_1, input_2], outputs=output)
   
    sgd = SGD(lr=0.01, momentum=0.9, decay=0.001)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model
    


proteinFeature_L = pd.read_csv("/home/jby2/XH/CellExch/dataset1/ligand_res_fea.csv", header=None, index_col=None)
proteinFeature_R = pd.read_csv("/home/jby2/XH/CellExch/dataset1/receptor_res_fea.csv", header=None, index_col=None)
interaction = pd.read_csv("/home/jby2/XH/CellExch/dataset1/ligand-receptor-interaction.csv", header=None, index_col=None, skiprows=1)  

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
row = sample.shape[0]
column = sample.shape[1]
index = [i for i in range(row)]
np.random.shuffle(index)
index = np.array(index)
sample = sample[index, :]
X = sample[:, 0: column-1]
y = sample[:, column-1:]
Trainlabels = y
scaler = StandardScaler().fit(X)
#scaler = RobustScaler().fit(X)
X = scaler.transform(X)

X1_train = X[:, :2813]
X2_train = X[:, 2813:]


##################################### Five-fold Cross-Validation ##########################################################
   
kf = StratifiedKFold(n_splits=5)

for train, test in kf.split(X, y):
    global model
    model=define_model()

    model.fit([X1_train[train],X2_train[train]],y[train],epochs=50,batch_size=64,verbose=1)
    y_test = y[test]
    y_score = model.predict([X1_train[test], X2_train[test]])
    
    
################################Intermediate Layer prediction (Abstraction features extraction)######################################
intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer('merged_protein1_2').output)
intermediate_output_p1 = intermediate_layer_model.predict([X1_train,X2_train])  
p_merge = pd.DataFrame(intermediate_output_p1)    
X_train_feat = pd.concat((p_merge,pd.DataFrame(pd.DataFrame(Trainlabels))),axis=1,ignore_index=True)

Train = X_train_feat
Train = Train.sample(frac=1)
X = Train.iloc[:,0:128].values
y = Train.iloc[:,128:].values

extracted_df = X_train_feat

scaler = RobustScaler()
X = scaler.fit_transform(X)


##################################### Five-fold Cross-Validation ##########################################################

acc = 0
precision = 0
recall = 0
f1 = 0
AUC = 0
AUPR = 0

kf = StratifiedKFold(n_splits=5)
fold = 0

for train, test in tqdm(kf.split(X,y), total=kf.get_n_splits(X)):
    fold = fold + 1
    feature_train, feature_test = X[train], X[test]
    target_train, target_test = y[train], y[test]
    
    clf = XGBClassifier(n_estimators=100)
    clf.fit(feature_train, target_train.ravel())
    
    y_ = clf.predict(feature_test)
    prob_output = clf.predict_proba(feature_test)
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
    directory = '/home/jby2/XH/CellExch/comparative_method_DNNXGB/roc/dataset1/'
    df.to_csv(directory + f"{fold}" + ".csv", index=False)
    
    data = {'Recall': rec, 'Precision': prec}
    df = pd.DataFrame(data)
    directory = '/home/jby2/XH/CellExch/comparative_method_DNNXGB/prc/dataset1/'
    df.to_csv(directory + f"{fold}" + ".csv", index=False)
    

acc = acc / 5
precision = precision / 5
recall = recall / 5
f1 = f1 / 5
AUC = AUC / 5
AUPR = AUPR / 5
with open("/home/jby2/XH/CellExch/comparative_method_DNNXGB/AUC_AUPR/dataset1/mean_value_of_five-fold_cross-validation.txt", mode="a") as f:
    f.write("precision:{}\n".format(precision))
    f.write("recall:{}\n".format(recall))
    f.write("acc:{}\n".format(acc))
    f.write("f1:{}\n".format(f1))
    f.write("AUC:{}\n".format(AUC))
    f.write("AUPR:{}\n\n".format(AUPR))


