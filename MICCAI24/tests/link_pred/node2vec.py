
import torch
import yaml
import torch_geometric.transforms as T
import random
import os
from torch.optim import Adam
from torch_geometric.transforms import RandomLinkSplit
from src.data import hemibrain
from src.data import data_loader
from src.models.GraphTasks.connection_predictor import ConnectionPredictor
from src.eval import evaluate_LinkPred
from src import train
from torch_geometric.nn import Node2Vec
import numpy as np
from sklearn.metrics import accuracy_score,roc_auc_score,recall_score,f1_score,auc,precision_recall_curve,roc_curve,precision_score
import random
import argparse
CURRENT_DEVICE = torch.cuda.current_device()
TOTAL_MEM = torch.cuda.get_device_properties(CURRENT_DEVICE).total_memory

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

    learning_rate = config['model']['learning_rate']
    batch_size = config['training']['batch_size']
    num_epochs = config['training']['epochs']
    hidden_dim = config['model']['hidden_size']
    device = config['model']['device']
    num_layers = config['model']['num_layers']
    path = config['data']['train_hemibrain_path']

    parser = argparse.ArgumentParser(description='Training configuration')
    parser.add_argument('--learning_rate', type=float, default=config['model']['learning_rate'])
    parser.add_argument('--batch_size', type=float, default=config['training']['batch_size'])
    parser.add_argument('--num_epochs', type=int, default=config['training']['epochs'])
    parser.add_argument('--hidden_dim', type=int, default=config['model']['hidden_size'])
    parser.add_argument('--device', type=str, default=config['model']['device'])
    parser.add_argument('--num_layers', type=int, default=config['model']['num_layers'])
    parser.add_argument('--path', type=str, default=config['data']['train_hemibrain_path'])
    args = parser.parse_args()
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    hidden_dim = args.hidden_size
    device = args.device
    num_layers = args.num_layers
    path = args.path
    setup_seed(0)

    transform = T.Compose([
            T.ToDevice(device),
            T.NormalizeFeatures(),
            RandomLinkSplit(num_val=0.85, num_test=0.1,
                            add_negative_train_samples=False),
        ])
    dataset = hemibrain.HemiBrain(path, transform=transform)
    train_data, val_data, test_data = dataset[0]
    model = Node2Vec(train_data.edge_index, embedding_dim=hidden_dim, walk_length=20,
                context_size=10, walks_per_node=10, num_negative_samples=1,
                sparse=True).to(device)
    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    for epoch in range(1, num_epochs):
        loss = train.train(model, criterion, train_data.x, train_data.edge_index, optimizer)

    with torch.no_grad():
        z = model(train_data.x, train_data.edge_index, None)

    train_loader = data_loader.LinkPred_Loader({'batch_size':1000,'device':device},train_data)
    config = {'input_channels':hidden_dim,'dropout':0.3}
    pair_deco = ConnectionPredictor(config,device=device)
    opt = torch.optim.Adam(pair_deco.parameters(),lr=0.003)
    link_loss = torch.nn.BCELoss()

    for ep in range(500):
        train.train_linkpred(pair_deco,train_loader,opt,z,link_loss)
    test_result = evaluate_LinkPred(pair_deco, test_data.test_data,z,device=device)
    print(test_result)

if __name__ == '__main__':
    main()
