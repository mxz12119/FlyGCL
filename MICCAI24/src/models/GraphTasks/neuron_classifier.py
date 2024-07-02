import torch
from torch.nn import Linear,Dropout,ReLU
class NeuronClassifier(torch.nn.Module):
    def __init__(self,config,device='cpu') -> None:
        super().__init__()
        self.mlp=torch.nn.Sequential(Linear(config['input_channels'],256,bias=True),Dropout(config['dropout'],inplace=True),ReLU(inplace=True),\
            Linear(256,config['out_channels'],bias=True))
        self.device=device
        self.mlp.to(device=self.device)
    def forward(self,x):
        x=self.mlp(x)
        return x