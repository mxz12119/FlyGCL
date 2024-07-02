import networkx as nx
import logging
import numpy as np
import os
import yaml
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import negative_sampling
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import accuracy_score,roc_auc_score,recall_score,f1_score,auc,precision_recall_curve,roc_curve,precision_score
import torch
from src.data import data_loader
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import math
import random
CURRENT_DEVICE = torch.cuda.current_device()
TOTAL_MEM = torch.cuda.get_device_properties(CURRENT_DEVICE).total_memory

def test(data,hash,device='cpu'):
    test_loader=data_loader.LinkPred_Loader({'device':device},data)
    y_pred_list,y_list=[],[]
    for batch in test_loader:
        test_data,pos,neg=batch
        test_edge=torch.cat([pos,neg],dim=1)
        y=torch.cat([torch.ones(size=(pos.size(1),),device=pos.device),torch.zeros(size=(neg.size(1),),device=neg.device)])
        y_pred=[]
        for i in range(len(test_edge[0])):
            u=test_edge[0][i].item()
            v=test_edge[1][i].item()
            if u!=v and u in hash and v in hash:
                W=hash[u].intersection(hash[v])
                res=[]
                for w in W:
                    if w in hash:
                        res.append(1 / math.log(len(hash[w])))
                num=sum(res)
            else:
                num= 0
            if(num>1):
                num=1
            y_pred.append(num)
        true_y = y.cpu().numpy()
        y_pred_list.append(y_pred)
        y_list.append(true_y)
    
    total_y_pred=np.concatenate(y_pred_list,axis=0)
    total_y=np.concatenate(y_list,axis=0)
    

    auc=roc_auc_score(total_y,total_y_pred)
    fpr,tpr,threshold = roc_curve(total_y, total_y_pred)
    yoden_index=np.argmax(tpr-fpr)
    tau=threshold[yoden_index]
    predictions=total_y_pred>tau
    predictions=predictions.astype(dtype=np.int32)
    return auc
def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic =True

def main():
    with open('/configs/model.yaml', 'r') as file:
        config = yaml.safe_load(file)
    path = config['data']['train_hemibrain_path']
    dataset=torch.load(path)


    setup_seed(0)
    transform = RandomLinkSplit(num_val=0.85, num_test=0.1, is_undirected=True)
    train_data, val_data, test_data = transform(dataset[0])

    edge_index=train_data.edge_index

    nodes=train_data.x.shape[0]
    G = nx.from_edgelist(edge_index.T.tolist())

    hash={}
    for i in G.nodes:
        hash[i]=set(list(G.neighbors(i)))
    result=test(test_data,hash)
    print(result)

    
if __name__ == '__main__':
    main()


