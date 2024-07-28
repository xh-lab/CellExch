import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from scipy import interp


# plot fig of the five comparative method and our method in dataset1
CB91_Blue = '#4370B4'
CB91_Green = '#549F9A'
CB91_Pink = '#C30078'
CB91_Orange = '#FD763F'
CB91_Pur = '#7C1A97'
CB91_Yellow = '#EFC57F'
colorlist = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Yellow, CB91_Orange, CB91_Pur]


directory_list = ["/home/jby2/XH/CellExch/dataset1/five-fold-cross-validation/roc/", "/home/jby2/XH/CellExch/comparative_method_lightgbm/roc/dataset1/", "/home/jby2/XH/CellExch/comparative_method_xgboost/roc/dataset1/", "/home/jby2/XH/CellExch/comparative_method_cellenboost/roc/dataset1/", "/home/jby2/XH/CellExch/comparative_method_DNNXGB/roc/dataset1/", "/home/jby2/XH/CellExch/comparative_method_celldialog/roc/dataset1/"]
              
label_list = ["CellExch (AUC = %0.4f)", "LightGBM (AUC = %0.4f)", "XGBoost (AUC = %0.4f)", "CellEnboost (AUC = %0.4f)", "DNN-XGBoost (AUC = %0.4f)", "CellDialog (AUC = %0.4f)"]


mean_fpr = np.linspace(0, 1, 1000)
i = 0

for directory in tqdm(directory_list, total=np.array(directory_list).shape[0]):
    tprs = []
    for fold in range(5):
        if i != 0:
            fold = fold + 1
        data = pd.read_csv(directory + f"{fold}" + ".csv")
        fpr = data['FPR'].tolist()
        tpr = data['TPR'].tolist()
        # roc_auc = auc(fpr, tpr)
        tprs.append(interp(mean_fpr, fpr, tpr))
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color=colorlist[i], label=label_list[i] % (mean_auc), lw=1.5, alpha=1)
    i = i + 1

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate',fontsize=13)
plt.ylabel('True Positive Rate',fontsize=13)
plt.title('ROC CURVE')
plt.legend(loc='lower right')
plt.savefig("/home/jby2/XH/CellExch/compare_roc_aupr/roc/dataset1/ROC-5fold_.pdf",dpi=1080)
plt.close()



# plot fig of the five comparative method and our method in dataset2
directory_list = ["/home/jby2/XH/CellExch/dataset2/five-fold-cross-validation/roc/", "/home/jby2/XH/CellExch/comparative_method_lightgbm/roc/dataset2/", "/home/jby2/XH/CellExch/comparative_method_xgboost/roc/dataset2/", "/home/jby2/XH/CellExch/comparative_method_cellenboost/roc/dataset2/", "/home/jby2/XH/CellExch/comparative_method_DNNXGB/roc/dataset2/", "/home/jby2/XH/CellExch/comparative_method_celldialog/roc/dataset2/"]
              

i = 0

for directory in tqdm(directory_list, total=np.array(directory_list).shape[0]):
    tprs = []
    for fold in range(5):
        if i != 0:
            fold = fold + 1
        data = pd.read_csv(directory + f"{fold}" + ".csv")
        fpr = data['FPR'].tolist()
        tpr = data['TPR'].tolist()
        # roc_auc = auc(fpr, tpr)
        tprs.append(interp(mean_fpr, fpr, tpr))
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color=colorlist[i], label=label_list[i] % (mean_auc), lw=1.5, alpha=1)
    i = i + 1

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate',fontsize=13)
plt.ylabel('True Positive Rate',fontsize=13)
plt.title('ROC CURVE')
plt.legend(loc='lower right')
plt.savefig("/home/jby2/XH/CellExch/compare_roc_aupr/roc/dataset2/ROC-5fold_.pdf",dpi=1080)
plt.close()



# plot fig of the five comparative method and our method in dataset3
directory_list = ["/home/jby2/XH/CellExch/dataset3/five-fold-cross-validation/roc/", "/home/jby2/XH/CellExch/comparative_method_lightgbm/roc/dataset3/", "/home/jby2/XH/CellExch/comparative_method_xgboost/roc/dataset3/", "/home/jby2/XH/CellExch/comparative_method_cellenboost/roc/dataset3/", "/home/jby2/XH/CellExch/comparative_method_DNNXGB/roc/dataset3/", "/home/jby2/XH/CellExch/comparative_method_celldialog/roc/dataset3/"]

i = 0

for directory in tqdm(directory_list, total=np.array(directory_list).shape[0]):
    tprs = []
    for fold in range(5):
        if i != 0:
            fold = fold + 1
        data = pd.read_csv(directory + f"{fold}" + ".csv")
        fpr = data['FPR'].tolist()
        tpr = data['TPR'].tolist()
        # roc_auc = auc(fpr, tpr)
        tprs.append(interp(mean_fpr, fpr, tpr))
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color=colorlist[i], label=label_list[i] % (mean_auc), lw=1.5, alpha=1)
    i = i + 1

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate',fontsize=13)
plt.ylabel('True Positive Rate',fontsize=13)
plt.title('ROC CURVE')
plt.legend(loc='lower right')
plt.savefig("/home/jby2/XH/CellExch/compare_roc_aupr/roc/dataset3/ROC-5fold_.pdf",dpi=1080)
plt.close()



# plot fig of the five comparative method and our method in dataset4
# mean_fpr = np.linspace(0, 1, 1000)
directory_list = ["/home/jby2/XH/CellExch/dataset4/five-fold-cross-validation/roc/", "/home/jby2/XH/CellExch/comparative_method_lightgbm/roc/dataset4/", "/home/jby2/XH/CellExch/comparative_method_xgboost/roc/dataset4/", "/home/jby2/XH/CellExch/comparative_method_cellenboost/roc/dataset4/", "/home/jby2/XH/CellExch/comparative_method_DNNXGB/roc/dataset4/", "/home/jby2/XH/CellExch/comparative_method_celldialog/roc/dataset4/"]

i = 0

for directory in tqdm(directory_list, total=np.array(directory_list).shape[0]):
    tprs = []
    for fold in range(5):
        if i != 0:
            fold = fold + 1
        data = pd.read_csv(directory + f"{fold}" + ".csv")
        fpr = data['FPR'].tolist()
        tpr = data['TPR'].tolist()
        # roc_auc = auc(fpr, tpr)
        tprs.append(interp(mean_fpr, fpr, tpr))
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color=colorlist[i], label=label_list[i] % (mean_auc), lw=1.5, alpha=1)
    i = i + 1

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate',fontsize=13)
plt.ylabel('True Positive Rate',fontsize=13)
plt.title('ROC CURVE')
plt.legend(loc='lower right')
plt.savefig("/home/jby2/XH/CellExch/compare_roc_aupr/roc/dataset4/ROC-5fold_.pdf",dpi=1080)
plt.close()




# dataset1
directory_list = ["/home/jby2/XH/CellExch/dataset1/five-fold-cross-validation/prc/", "/home/jby2/XH/CellExch/comparative_method_lightgbm/prc/dataset1/", "/home/jby2/XH/CellExch/comparative_method_xgboost/prc/dataset1/", "/home/jby2/XH/CellExch/comparative_method_cellenboost/prc/dataset1/", "/home/jby2/XH/CellExch/comparative_method_DNNXGB/prc/dataset1/", "/home/jby2/XH/CellExch/comparative_method_celldialog/prc/dataset1/"]
label_list = ["CellExch (AUPR = %0.4f)", "LightGBM (AUPR = %0.4f)", "XGBoost (AUPR = %0.4f)", "CellEnboost (AUPR = %0.4f)", "DNN-XGBoost (AUPR = %0.4f)", "CellDialog (AUPR = %0.4f)"]

mean_precision = np.linspace(0, 1, 1000)
i = 0

for directory in tqdm(directory_list, total=np.array(directory_list).shape[0]):
    recalls = []
    for fold in range(5):
        if i != 0:
            fold = fold + 1
        data = pd.read_csv(directory + f"{fold}" + ".csv")
        precision = data['Precision'].tolist()
        recall = data['Recall'].tolist()
        # roc_auc = auc(fpr, tpr)
        recalls.append(interp(mean_precision, precision, recall))
    
    mean_recall = np.mean(recalls, axis=0)
    mean_aupr = auc(mean_recall, mean_precision)
    plt.plot(mean_recall, mean_precision, color=colorlist[i], label=label_list[i] % (mean_aupr), lw=1.5, alpha=1)
    i = i + 1

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Recall',fontsize=13)
plt.ylabel('Precision',fontsize=13)
plt.title('PR CURVE')
plt.legend(loc='lower left')
plt.savefig("/home/jby2/XH/CellExch/compare_roc_aupr/prc/dataset1/PRC-5fold__.pdf",dpi=1080)
plt.close()


# dataset2
directory_list = ["/home/jby2/XH/CellExch/dataset2/five-fold-cross-validation/prc/", "/home/jby2/XH/CellExch/comparative_method_lightgbm/prc/dataset2/", "/home/jby2/XH/CellExch/comparative_method_xgboost/prc/dataset2/", "/home/jby2/XH/CellExch/comparative_method_cellenboost/prc/dataset2/", "/home/jby2/XH/CellExch/comparative_method_DNNXGB/prc/dataset2/", "/home/jby2/XH/CellExch/comparative_method_celldialog/prc/dataset2/"]

i = 0

for directory in tqdm(directory_list, total=np.array(directory_list).shape[0]):
    recalls = []
    for fold in range(5):
        if i != 0:
            fold = fold + 1
        data = pd.read_csv(directory + f"{fold}" + ".csv")
        precision = data['Precision'].tolist()
        recall = data['Recall'].tolist()
        recalls.append(interp(mean_precision, precision, recall))
    
    mean_recall = np.mean(recalls, axis=0)
    mean_aupr = auc(mean_recall, mean_precision)
    plt.plot(mean_recall, mean_precision, color=colorlist[i], label=label_list[i] % (mean_aupr), lw=1.5, alpha=1)
    i = i + 1

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Recall',fontsize=13)
plt.ylabel('Precision',fontsize=13)
plt.title('PR CURVE')
plt.legend(loc='lower left')
plt.savefig("/home/jby2/XH/CellExch/compare_roc_aupr/prc/dataset2/PRC-5fold__.pdf",dpi=1080)
plt.close()


# dataset3
directory_list = ["/home/jby2/XH/CellExch/dataset3/five-fold-cross-validation/prc/", "/home/jby2/XH/CellExch/comparative_method_lightgbm/prc/dataset3/", "/home/jby2/XH/CellExch/comparative_method_xgboost/prc/dataset3/", "/home/jby2/XH/CellExch/comparative_method_cellenboost/prc/dataset3/", "/home/jby2/XH/CellExch/comparative_method_DNNXGB/prc/dataset3/", "/home/jby2/XH/CellExch/comparative_method_celldialog/prc/dataset3/"]
              
i = 0

for directory in tqdm(directory_list, total=np.array(directory_list).shape[0]):
    recalls = []
    for fold in range(5):
        if i != 0:
            fold = fold + 1
        data = pd.read_csv(directory + f"{fold}" + ".csv")
        precision = data['Precision'].tolist()
        recall = data['Recall'].tolist()
        recalls.append(interp(mean_precision, precision, recall))
    
    mean_recall = np.mean(recalls, axis=0)
    mean_aupr = auc(mean_recall, mean_precision)
    plt.plot(mean_recall, mean_precision, color=colorlist[i], label=label_list[i] % (mean_aupr), lw=1.5, alpha=1)
    i = i + 1

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Recall',fontsize=13)
plt.ylabel('Precision',fontsize=13)
plt.title('PR CURVE')
plt.legend(loc='lower left')
plt.savefig("/home/jby2/XH/CellExch/compare_roc_aupr/prc/dataset3/PRC-5fold__.pdf",dpi=1080)
plt.close()


# dataset4
directory_list = ["/home/jby2/XH/CellExch/dataset4/five-fold-cross-validation/prc/", "/home/jby2/XH/CellExch/comparative_method_lightgbm/prc/dataset4/", "/home/jby2/XH/CellExch/comparative_method_xgboost/prc/dataset4/", "/home/jby2/XH/CellExch/comparative_method_cellenboost/prc/dataset4/", "/home/jby2/XH/CellExch/comparative_method_DNNXGB/prc/dataset4/", "/home/jby2/XH/CellExch/comparative_method_celldialog/prc/dataset4/"]
              
i = 0

for directory in tqdm(directory_list, total=np.array(directory_list).shape[0]):
    recalls = []
    for fold in range(5):
        if i != 0:
            fold = fold + 1
        data = pd.read_csv(directory + f"{fold}" + ".csv")
        precision = data['Precision'].tolist()
        recall = data['Recall'].tolist()
        # roc_auc = auc(fpr, tpr)
        recalls.append(interp(mean_precision, precision, recall))
    
    mean_recall = np.mean(recalls, axis=0)
    mean_aupr = auc(mean_recall, mean_precision)
    plt.plot(mean_recall, mean_precision, color=colorlist[i], label=label_list[i] % (mean_aupr), lw=1.5, alpha=1)
    i = i + 1

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Recall',fontsize=13)
plt.ylabel('Precision',fontsize=13)
plt.title('PR CURVE')
plt.legend(loc='lower left')
plt.savefig("/home/jby2/XH/CellExch/compare_roc_aupr/prc/dataset4/PRC-5fold__.pdf",dpi=1080)
plt.close()













