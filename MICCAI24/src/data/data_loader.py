import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import negative_sampling
CURRENT_DEVICE = torch.cuda.current_device()
TOTAL_MEM = torch.cuda.get_device_properties(CURRENT_DEVICE).total_memory

class LinkPred_Loader:
    def __init__(self,config,dataset: InMemoryDataset):
        self.dataset=dataset
        self.loader_config=config
        self.device=config['device']
        if 'negtive_num' in self.loader_config:
            self.negtive_num=self.loader_config['negtive_num']
        else:
            self.negtive_num=1
        
        self.batch_size=self.loader_config.get('batch_size',False)

        if self.batch_size is False:
            if dataset.edge_index.size(1)*(self.negtive_num+1)*512<TOTAL_MEM/8:
                self.batch_size=dataset.edge_index.size(1)
            else:
                self.batch_size=min(5000,dataset.edge_index.size(1))
        self.step=0
    def __iter__(self):
        self.step=0
        self.pos_edge_index=self.dataset.edge_index
        pos_perm=torch.randperm(self.pos_edge_index.size(1),device=self.device)
        pos_batch_index=torch.split(pos_perm,self.batch_size)
        self.neg_edge_index = negative_sampling(
            edge_index=self.dataset.edge_index, num_nodes=self.dataset.num_nodes,
            num_neg_samples=self.dataset.edge_index.size(1)*self.negtive_num, method='sparse')
        
        neg_perm = torch.randperm(self.neg_edge_index.size(1),device=self.device)
        neg_batch_index=torch.split(neg_perm,self.batch_size*self.negtive_num)
        assert len(neg_batch_index)==len(pos_batch_index)
        self.batch_index=list(zip(pos_batch_index,neg_batch_index))

        self.pos_edge_index=self.pos_edge_index.T
        self.neg_edge_index=self.neg_edge_index.T
        return self
    def __next__(self):
        if self.step>=len(self.batch_index):
            raise StopIteration
        else:
            pos_ind,neg_ind=self.batch_index[self.step]

            batch_pos_edge_index=self.pos_edge_index[pos_ind]
            
            pos=batch_pos_edge_index.T

            batch_neg_edge_index=self.neg_edge_index[neg_ind]
            
            neg=batch_neg_edge_index.T
            
            batch=(self.dataset,pos,neg)
            self.step+=1
        return batch
    def __len__(self):
        return len(self.batch_index)


class NeuronClassDataLoader:
    def __init__(self, config, dataset: InMemoryDataset, z):
        self.dataset = dataset
        self.loader_config = config
        self.device = config['device']
        self.batch_size = config.get('batch_size', 64)  
        self.step = 0
        self.z = z

    def __iter__(self):
        self.step = 0

        if self.batch_size > self.dataset.num_nodes:
            self.batch_size = self.dataset.num_nodes

        self.batch_indices = torch.randperm(self.dataset.num_nodes, device=self.device).split(self.batch_size)

        return self

    def __next__(self):
        if self.step >= len(self.batch_indices):
            raise StopIteration
        else:
            batch_nodes = self.batch_indices[self.step]
            batch = (self.z[batch_nodes], self.dataset.y.T[batch_nodes])
            self.step += 1
        return batch

    def __len__(self):
        return len(self.batch_indices)
