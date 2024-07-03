import torch
import yaml
import numpy as np
from models.GraphEncoder import gcn,gin,graphsage,flygcl
import torch_geometric.transforms as T
from torch_geometric.transforms import RandomLinkSplit
from data import hemibrain,manc,data_loader
from torch.optim import Adam
from torch.functional import F
import GCL.losses as L
import GCL.augmentors as A
from tqdm import tqdm
from GCL.models import DualBranchContrast
from models.GraphTasks.connection_predictor import ConnectionPredictor
from models.GraphTasks.neuron_classifier import NeuronClassifier
from sklearn.metrics import accuracy_score,roc_auc_score,recall_score,f1_score,auc,precision_recall_curve,roc_curve,precision_score,classification_report
class Model():
    def __init__(self,yaml_path):
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
        self.modelname = data.get('model')['name']
        self.lr = data.get('model')['learning_rate']
        self.hidden_size = data.get('model')['hidden_size']
        self.num_layers = data.get('model')['num_layers']
        self.epochs = data.get("training")['epochs']
        self.batch_size = data.get('training')['batch_size']
        dataset = data.get('dataset')
        self.task = data.get('task')
        self.path = data.get('path')
        self.device = data.get('device')
        transform = T.Compose([
            T.ToDevice(self.device),
            T.NormalizeFeatures(),
            RandomLinkSplit(num_val=0.85, num_test=0.1,
                            add_negative_train_samples=False),
        ])
        #dataset load
        if(dataset.upper() == 'HEMIBRAIN'):
            self.dataset = hemibrain.HemiBrain(self.path,transform=transform)
        if(dataset.upper() == 'MANC'):
            self.dataset = manc.Manc(self.path,transform=transform)
        self.train_data,self.val_data,self.test_data = self.dataset[0]
        #model load

        if(self.task.upper() == "NEURON_CLASSIFICATION"):
            out_channels = torch.max(self.dataset.y)+1
        if(self.task.upper() == "LINK_PREDICTION"):
            out_channels = self.hidden_size
        if(self.modelname.upper() == 'GCN'):
            self.model = gcn.GCN(in_channels=self.dataset.num_features, hidden_channels=self.hidden_size, out_channels=out_channels, node_num=self.dataset[0][0].x.shape[0], num_layers=self.num_layers)
        if(self.modelname.upper() == 'GIN'):
            self.model = gin.GIN(in_channels=self.dataset.num_features, hidden_channels=self.hidden_size, out_channels=out_channels, node_num=self.dataset[0][0].x.shape[0], num_layers=self.num_layers)
        if(self.modelname.upper() == 'GRAPHSAGE'):
            self.model = graphsage.GraphSAGE(in_channels=self.dataset.num_features, hidden_channels=self.hidden_size, out_channels=out_channels, node_num=self.dataset[0][0].x.shape[0], num_layers=self.num_layers)
        if(self.modelname.upper() == 'FLYGCL'):
            aug1 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])
            aug2 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])
            gconv = flygcl.GConv(input_dim=self.dataset.num_features, hidden_dim=self.hidden_size, activation=torch.nn.ReLU, num_layers=self.num_layers, node_num = self.dataset[0][0].x.shape[0]).to(self.device)
            self.model = flygcl.Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=self.hidden_size, proj_dim=32).to(self.device)

    def train(self):
        if(self.task.upper() == "NEURON_CLASSIFICATION"):
            self.train1()
        if(self.task.upper() == "LINK_PREDICTION"):
            self.train2()
    def train1(self):
        if(self.modelname.upper() in ['GCN','GIN','GRAPHSAGE']):
            model = self.model
            optimizer = Adam(model.parameters(), lr=self.lr)
            for ep in range(self.epochs):
                model.train()
                optimizer.zero_grad()
                z = model(self.train_data.edge_index)
                loss = F.cross_entropy(z,self.train_data.y)
                loss.backward()
                optimizer.step()

                print('epochs:',ep ,' train_loss:',loss)
                if ep % 10==0:
                    with torch.no_grad():
                        model.eval()
                        z = model(self.test_data.edge_index)
                        valid_loss = F.cross_entropy(z, self.test_data.y)
                    print('epochs:',self.epochs,'valid loss:',valid_loss.item())
        if(self.modelname.upper() in ['FLYGCL']):
            model = self.model
            contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=True).to(self.device)
            optimizer = Adam(model.parameters(), lr=0.01)

            for epoch in range(1, self.epochs):
                model.train()
                optimizer.zero_grad()
                z, z1, z2 = model(self.train_data.x, self.train_data.edge_index, None)
                h1, h2 = [x for x in [z1, z2]]
                loss = contrast_model(h1, h2)
                loss.backward()
                optimizer.step()
                if epoch % 10==0:
                    with torch.no_grad():
                        model.eval()
                        z, z1, z2 = model(self.test_data.x, self.test_data.edge_index, None)
                        h1, h2 = [model.project(z) for x in [z1, z2]]
                        valid_loss = contrast_model(h1, h2)
                        
                    print('contrastive learning valid loss:',valid_loss.item(),'epoch:',epoch)
                    print('contrastive learning training loss:',loss,'epoch:',epoch)
            with torch.no_grad():
                z, _, _ = model(self.train_data.x, self.train_data.edge_index, None)
            config={'input_channels':self.hidden_size,'dropout':0.3,'out_channels':torch.max(self.train_data.y)+1}
            model=NeuronClassifier(config,device=self.device)

            optimizer = Adam(model.parameters(), lr=0.01)
            cross_loss = torch.nn.CrossEntropyLoss()
            z=z.to(self.device)
            ratio=0.1
            train_num = int(self.train_data.x.shape[0]*ratio)
            for i in range(200):
                optimizer.zero_grad()
                output=model(z)
                loss=cross_loss(output[:train_num],self.train_data.y[:train_num])
                loss.backward()
                optimizer.step()

            output=model(z[train_num:])
            data = [torch.argmax(i).cpu().item() for i in output]

            acc=accuracy_score(data,self.test_data.y[train_num:].cpu())
            f1_macro=f1_score(data,self.test_data.y[train_num:].cpu(),average="macro")
            f1_micro=f1_score(data,self.test_data.y[train_num:].cpu(),average="micro")
 
            print('f1_macro',f1_macro)
            print('f1_micro',f1_micro)
    def train2(self):
        if(self.modelname.upper() in ['GCN','GIN','GRAPHSAGE']):
            train_loader=data_loader.LinkPred_Loader({'batch_size':self.batch_size,'device':self.device,'shuffle':True},self.train_data)
            test_loader=data_loader.LinkPred_Loader({'batch_size':self.batch_size,'device':self.device,'shuffle':True},self.test_data)
            model = self.model
            opt=torch.optim.Adam(model.parameters(),lr = self.lr)
            criterion = torch.nn.BCEWithLogitsLoss()

            for ep in range(self.epochs):
                total_loss=0
                for batch in train_loader:
                    model.train()
                    opt.zero_grad()
                    _,pos,neg=batch
                    edge=torch.cat([pos,neg],dim=1)
                    z = model(edge)
                    y_pred = (z[edge[0]]*z[edge[1]]).sum(dim=-1).view(-1)
                    y=torch.cat([torch.ones(size=(pos.size(1),),device=pos.device),torch.zeros(size=(neg.size(1),),device=neg.device)])
                    loss=criterion(y_pred,y)
                    loss.backward()
                    opt.step()
                    total_loss+=loss.detach().item()

                print('epochs:',ep ,' train_loss:',total_loss)
                if ep % 10==100:
                    with torch.no_grad():
                        model.eval()
                        valid_loss=0
                        for batch in test_loader:
                            _,pos,neg=batch
                            edge=torch.cat([pos,neg],dim=1)
                            z = model(edge)
                            y_pred = (z[edge[0]]*z[edge[1]]).sum(dim=-1)
                            y=torch.cat([torch.ones(size=(pos.size(1),),device=pos.device),torch.zeros(size=(neg.size(1),),device=neg.device)])
                            loss=criterion(y_pred,y)
                            valid_loss+=loss.detach().item()
                    print('epochs:',self.epochs,'valid loss:',valid_loss)
        if(self.modelname.upper() in ['FLYGCL']):
            model = self.model
            contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=True).to(self.device)
            optimizer = Adam(model.parameters(), lr=0.001)

            for epoch in range(1, self.epochs):
                model.train()
                optimizer.zero_grad()
                z, z1, z2 = model(self.train_data.x, self.train_data.edge_index, None)
                h1, h2 = [x for x in [z1, z2]]
                loss = contrast_model(h1, h2)
                loss.backward()
                optimizer.step()
                print('epochs:',epoch,'loss:',loss.item())
                if epoch % 10==0:
                    with torch.no_grad():
                        model.eval()
                        z, z1, z2 = model(self.test_data.x, self.test_data.edge_index, None)
                        h1, h2 = [model.project(z) for x in [z1, z2]]
                        valid_loss = contrast_model(h1, h2)
                        
                    print('contrastive learning valid loss:',valid_loss.item(),'epoch:',epoch)
                    print('contrastive learning training loss:',loss,'epoch:',epoch)
            with torch.no_grad():
                z, _, _ = model(self.train_data.x, self.train_data.edge_index, None)
            train_loader=data_loader.LinkPred_Loader({'batch_size':self.batch_size,'device':self.device,'shuffle':True},self.train_data)
            config={'input_channels':self.hidden_size,'dropout':0.3}
            pair_deco=ConnectionPredictor(config,device=self.device)
            opt=torch.optim.Adam(pair_deco.parameters(),lr=0.003)
            link_loss=torch.nn.BCELoss()

            for ep in range(200):
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
                print('pair decoder loss:',total_loss,'epoch:',ep)
                # if ep %10 == 0:
                #     print('pair decoder loss:',total_loss,'epoch:',ep)


            test_result = self.test(pair_deco, self.test_data,z,device=self.device)
            print(test_result)
    def save(self):
        pass
    def test(self,deco,data,z,device='cpu'):
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
        

        return result
m = Model("project/MICCAI24/configs/model.yaml")
print(m.model)

m.train()