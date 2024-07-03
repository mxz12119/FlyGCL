import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, node_num, num_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.x = torch.nn.Parameter(
                torch.empty(size=(node_num, in_channels), dtype=torch.float32),
                requires_grad=True)
        self.layers.append(SAGEConv(in_channels, hidden_channels, cached=False))
    
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_channels, hidden_channels, cached=False))
        self.layers.append(SAGEConv(hidden_channels, out_channels, cached=False))

    def forward(self, x, edge_index):
        x = self.x
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
    
        return x
