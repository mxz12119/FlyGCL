import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import GCL.losses as L
import GCL.augmentors as A
from GCL.models import DualBranchContrast
from torch.optim import Adam
from data import data_loader

class GConv(torch.nn.Module):
    def __init__(self, input_dim, node_num, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.x = torch.nn.Parameter(
                torch.rand(size=(node_num, input_dim), dtype=torch.float32),
                requires_grad=True)
        self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))
    
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))

    def forward(self, edge_index, edge_weight=None):
        z = self.x
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
            
        return z

class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

        self.fc1 = torch.nn.Linear(hidden_dim,hidden_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z = self.encoder(edge_index, edge_weight)
        z1 = self.encoder(edge_index1, edge_weight1)
        z2 = self.encoder(edge_index2, edge_weight2)
        return z, z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return z
    
