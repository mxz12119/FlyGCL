from hashlib import new
from typing import Optional, Callable, List
import os.path as osp
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data
import networkx as nx
import pandas as pd


class Manc(InMemoryDataset):
    url=None
    r"""
    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'null'

    def __init__(self, root: str,split='node', transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,process=False,restore_split=None,device=None):

        super().__init__(root, transform, pre_transform)
        if device is None:
            self.device='cpu'
        else:
            self.device=device
        if process:
            self._process()
        if restore_split is None:
            self._data, self.slices = torch.load(self.processed_paths[0],map_location=self.device)
        elif isinstance(restore_split,str):
            self._data= torch.load(restore_split,map_location=self.device)
        self._data, self.slices = self.collate([self._data])

    @property
    def num_edges(self):
        return self._data.edge_index.size(1)
    @property
    def edge_index(self):
        return self._data.edge_index
    @property
    def y(self):
        return self._data.y
    
    @property
    def num_nodes(self):
        return self._data.num_nodes
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')
    @property
    def raw_file_names(self) -> List[str]:
        return ['connections-per-roi.csv','connections2.csv','neurons.csv']
    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
    @property
    def neu2ID(self):
        if hasattr(self,'neu2ID_dict'):
            return self.neu2ID_dict
        else:
            res=pd.read_csv(osp.join(self.raw_dir,'neurons.csv'))   
            self.neu2ID_dict=dict(zip(res['bodyId'],list(range(len(res['bodyId'])))))
            return self.neu2ID_dict
    @property
    def label2ID(self):
        if hasattr(self,'label2ID_dict'):
            return self.label2ID_dict
        else:
            res=pd.read_csv(osp.join(self.raw_dir,'neurons.csv'))
            label_set= set(res['type'].to_list())
            self.label2ID_dict={l:k for k,l in enumerate(label_set)}
            return self.label2ID_dict
    @property
    def neuron2label(self):
        if hasattr(self,'neuron2label_dict'):
            return self.neuron2label_dict
        else:
            
            res=pd.read_csv(osp.join(self.raw_dir,'neurons.csv'))   
            self.neuron2label_dict=dict(zip(res['bodyId'],res['type']))
            return self.neuron2label_dict
    @property
    def typedim(self):
        if hasattr(self,'cache_type_dim'):
            return self.cache_type_dim
        else:
            label2ID=self.label2ID
            self.cache_type_dim=len(label2ID.keys())
            return self.cache_type_dim

    @staticmethod
    def read_data(raw_dir):
        res=pd.read_csv(osp.join(raw_dir,'neurons.csv')) 
        neu2ID=dict(zip(res['bodyId'],list(range(len(res['bodyId'])))))
        ID2neu={v:k for k,v in neu2ID.items()}
       
        connections=pd.read_csv(osp.join(raw_dir,'connections2.csv'))
        
        edge_index=list(zip(connections['bodyId_pre'].to_list(),connections['bodyId_post'].to_list()))
        edge_attr=connections['weight'].to_list()
       
        edge_index=torch.Tensor(edge_index)
        edge_index=torch.transpose(edge_index,0,1)
        edge_index=edge_index.to(dtype=torch.int64)
        edge_attr=torch.Tensor(edge_attr)
        neuron2label=dict(zip(res['bodyId'],res['type']))
        label_set= set(neuron2label.values())
        label2ID={l:k for k,l in enumerate(label_set)}

        y=[]
        for i_neuron in range(len(ID2neu)):
            label_x=neuron2label[ID2neu[i_neuron]]
            y.append(label2ID[label_x])
        y=torch.tensor(y)

        x=torch.nn.functional.one_hot(y)
        x=x.to(dtype=torch.float32)

        return Data(x=x,edge_index=edge_index,y=y,edge_attr=edge_attr,neu2ID=neu2ID,label2ID=label2ID)

    def process(self):
        data = self.read_data(self.raw_dir)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])
    def manual_split(self,path):
        res=pd.read_csv(osp.join(self.raw_dir,'neurons.csv')) 
        neuron2label=dict(zip(res['bodyId'],res['type']))
        label_set= set(neuron2label.values())
        label2ID={l:k for k,l in enumerate(label_set)}
        neu2ID=dict(zip(res['bodyId'],list(range(len(res['bodyId'])))))
        
        for k,v in self.neu2ID.items():
            if neu2ID[k]!=v:
                raise ValueError('this file doesn`t match this object(%s)'%__name__)
        raise NotImplemented
    def __repr__(self) -> str:
        return 'Manc'
