import argparse
import torch
import os
import time
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from torch_geometric.data import Data
from torch.optim import Adam
import yaml
from .utils import dict_sequential
from tensorboardX import SummaryWriter

class Base_Trainer:
    def __init__(self,config,model,logdir=None):
        self.trainer_config= config
        self.model=model

        self.loss_function=self.get_loss_function(self.trainer_config['loss'])
        self.iter=0
        self.eval_indicator=[]
        self.parameters=[p for p in self.model.parameters() if p.requires_grad]
        self.optimizer=self.get_optimizer(self.trainer_config['optimizer'],self.parameters,self.trainer_config['lr'],self.trainer_config['weight_decay'])
        self.result_process=[]
        self.logdir=self.get_log_dir(logdir)
        self.best_result=None
        self.train_graph=None
        self.current_result=None

        self.writer=SummaryWriter(logdir=self.logdir) if self.trainer_config['use_tensor_board'] else None
    def get_log_dir(self,logdir=None):
        if logdir is None:
            import socket
            from datetime import datetime
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            logdir = os.path.join(
                'runs', self.trainer_config['task_save'],current_time + '_' + socket.gethostname())
        return logdir
    def reset(self):
        self.model.reset()
        self.iter=0
        self.optimizer = self.get_optimizer(self.trainer_config['optimizer'], self.parameters, self.trainer_config['lr'],
                                            self.trainer_config['weight_decay'])
    def update(self,*args,**kwargs):
        '''template method'''
        raise NotImplemented('update is not implemented')
    def get_loss_function(self,loss_name,**kwargs):
        if loss_name=='bcelogits':
            loss=torch.nn.BCEWithLogitsLoss(**kwargs)
        elif loss_name=='bce':
            loss=torch.nn.BCELoss(**kwargs)
        elif loss_name=='nll':
            loss=torch.nn.NLLLoss(**kwargs)
        elif loss_name=='cro_en':
            loss=torch.nn.CrossEntropyLoss(**kwargs)
        elif loss_name=='mse':
            loss=torch.nn.MSELoss(**kwargs)
        else:
            raise NotImplemented('%s is not implemented'%loss_name)
        return loss
    @staticmethod
    def get_optimizer(name, parameters, lr, weight_decay=0):
        if name == 'sgd':
            return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
        elif name == 'rmsprop':
            return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
        elif name == 'adagrad':
            return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
        elif name == 'adam':
            return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        elif name == 'adamax':
            return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
        else:
            raise Exception("Unsupported optimizer: {}".format(name))
    def excecute_epoch(self,train_loader,**kargs):
        for ep in range(self.trainer_config['epoch']):
            t0=time.time()
            train_result={'Training_Loss':0,'ep':ep}
            for k,batch in enumerate(train_loader):

                loss_result=self.update(batch)
                if self.writer:
                    self.writer.add_scalar('Training Loss',loss_result,global_step=self.iter)
                train_result['Training_Loss']+=loss_result
                self.iter +=1
            train_result['Training_Loss']=train_result['Training_Loss']/len(train_loader)
            train_result['time_cost']=time.time()-t0
            self.logging(train_result,'train')

           
        with open(os.path.join(self.logdir,'final_result.yaml'),'w') as fin:
            self.current_result.pop('ROC')
            self.current_result.pop('PR-curve')
            yaml.dump(dict_sequential(self.current_result),fin)
        torch.save(self.model,os.path.join(self.logdir,'checkpoint_final_model.pt'))

def train_encoder(encoder_model, contrast_model, x,edge_index, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z, z1, z2 = encoder_model(x, edge_index, None)
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
    loss = contrast_model(h1, h2)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_linkpred(pair_deco,train_loader,opt,z,link_loss):
    total_loss=0
    pair_deco.train()
    for batch in train_loader:
        opt.zero_grad()
        train_data,pos,neg=batch
        edge=torch.cat([pos,neg],dim=1)
        h_s,h_t=z[edge[0]],z[edge[1]]
        y_pred=pair_deco(h_s,h_t)
        y=torch.cat([torch.ones(size=(pos.size(1),),device=pos.device),torch.zeros(size=(neg.size(1),),device=neg.device)])
        loss=link_loss(y_pred,y)
        loss.backward()
        opt.step()
        total_loss+=loss.detach().item()

def train_classify(pair_deco,train_loader,opt,z,loss):
    total_loss=0
    pair_deco.train()
    for batch in train_loader:
        opt.zero_grad()
        test_x,test_y=batch
        y_pred=pair_deco(test_x)
        loss=loss(y_pred,test_y)
        loss.backward()
        opt.step()
        total_loss+=loss.detach().item()