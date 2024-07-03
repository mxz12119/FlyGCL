import torch
from torch.nn import Linear,Dropout,ReLU
class ConnectionPredictor(torch.nn.Module):
    def __init__(self,config,device='cpu') -> None:
        super().__init__()
        self.mlp=torch.nn.Sequential(Linear(config['input_channels']*2,256,bias=True),Dropout(config['dropout'],inplace=True),ReLU(inplace=True),\
            Linear(256,1,bias=True))
        self.device=device
        self.mlp.to(device=self.device)
    def forward(self,e1,e2):
        x=torch.cat((e1,e2),1)
        x=self.mlp(x)
        x=torch.sigmoid(x).view(-1)
        return x