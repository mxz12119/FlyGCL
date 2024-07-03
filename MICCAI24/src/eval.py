import torch
import numpy as np
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,f1_score,precision_score,recall_score,precision_recall_curve
from torch.utils.data import DataLoader
from src.data import data_loader

def evaluate_NeuClassify(deco,data,z,device='cpu'):
    deco.eval()
    test_loader=data_loader.NeuronClassDataLoader({'device':device},data)
    y_pred_list,y_list=[],[]
    for batch in test_loader:
        test_x,test_y=batch
        y_pred=deco(test_x)
        true_y,outcpu=test_y.cpu().numpy(),y_pred.detach().cpu().numpy()
        y_pred_list.append(outcpu)
        y_list.append(true_y)

    total_y_pred=np.concatenate(y_pred_list,axis=0)
    total_y=np.concatenate(y_list,axis=0)

    acc=accuracy_score(total_y,total_y_pred)
    f1_micro=f1_score(total_y,total_y_pred,"micro")
    f1_macro=f1_score(total_y,total_y_pred,"macro")
    result={'Accuracy':acc,'F1_Micro':f1_micro,'F1_Macro':f1_macro}

def evaluate_LinkPred(deco,data,z,device='cpu'):
    deco.eval()
    test_loader=data_loader.LinkPred_Loader({'device':device},data)
    y_pred_list,y_list=[],[]
    for batch in test_loader:
        test_data,pos,neg=batch
        test_edge=torch.cat([pos,neg],dim=1)

        y=torch.cat([torch.ones(size=(pos.size(1),),device=pos.device),torch.zeros(size=(neg.size(1),),device=neg.device)])
       
        h_s,h_t=z[test_edge[0]],z[test_edge[1]]
        y_pred=deco(h_s,h_t)
        true_y,outcpu=y.cpu().numpy(),y_pred.detach().cpu().numpy()
        y_pred_list.append(outcpu)
        y_list.append(true_y)

    total_y_pred=np.concatenate(y_pred_list,axis=0)
    total_y=np.concatenate(y_list,axis=0)

    auc=roc_auc_score(total_y,total_y_pred)
    fpr,tpr,threshold = roc_curve(total_y, total_y_pred)
    yoden_index=np.argmax(tpr-fpr)
    tau=threshold[yoden_index]
    predictions=total_y_pred>tau
    predictions=predictions.astype(dtype=np.int32)

    acc=accuracy_score(total_y,predictions)
    f1=f1_score(total_y,predictions)
    pre=precision_score(total_y,predictions)
    rec=recall_score(total_y,predictions)
    ROC_curve=roc_curve(total_y,total_y_pred)
    PR_curve=precision_recall_curve(total_y,total_y_pred)
    result={'Accuracy':acc,'Precision':pre,'Recall':rec,'F1':f1,'AUC':auc,'ROC':ROC_curve,'PR-curve':PR_curve,'tau':tau}
