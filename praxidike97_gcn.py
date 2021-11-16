#!~/miniconda3/envs/torch-env/bin/python
# https://github.com/praxidike97/GraphNeuralNet/blob/master/main.py
# To run on command line:
# CUDA_VISIBLE_DEVICES=0,1 CUDA_LAUNCH_BLOCKING=1 python praxidike97_gcn.py 
# python -m torch.distributed.run praxidike97_gcn.py
# GLOO_SOCKET_IFNAME=eno2 python -m torch.distributed.launch praxidike97_gcn.py
# python -m torch.distributed.launch --nproc_per_node=2 --nnode=2 --node_rank=0 praxidike97_gcn.py
# Check running processes: ps -elf | grep python
# Note: cannot run on slurm since can't activate conda environment

from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from torch.distributed.elastic.multiprocessing.errors import record
@record

class Net(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__() #Net, self
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


def plot_dataset(dataset):
    edges_raw = dataset.data.edge_index.numpy()
    edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]
    labels = dataset.data.y.numpy()

    G = nx.Graph()
    G.add_nodes_from(list(range(np.max(edges_raw))))
    G.add_edges_from(edges)
    plt.subplot(111)
    options = {
                'node_size': 30,
                'width': 0.2,
    }
    nx.draw(G, with_labels=False, node_color=labels.tolist(), cmap=plt.cm.tab10, font_weight='bold', **options)
    plt.show()

def test(model, data, train=True):
    model.eval()

    correct = 0
    pred = model(data).max(dim=1)[1]

    if train:
        correct += pred[data.train_mask].eq(data.y[data.train_mask]).sum().item()
        return correct / (len(data.y[data.train_mask]))
    else:
        correct += pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        return correct / (len(data.y[data.test_mask]))

def train(rank, world_size):
    torch.cuda.empty_cache()
    print("GPUs", torch.cuda.device_count())
    print("GLOO", os.environ.get('GLOO_SOCKET_IFNAME'))
    print("I am processor rank", rank)
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # check device
    #print(device)
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print(t, r, a, f)

    # dataset
    plot=False
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    #data_loader = DataLoader(dataset=dataset, batch_size = 30, shuffle=True, num_workers=4, pin_memory=False)
    data = dataset[0].to(rank)

    # create default process group
    dist.init_process_group(backend="nccl")#rank=rank and world_size=world_size are automatically set

    # create local model
    model = Net(dataset).to(rank)
    
    # construct distributed data parallel model
    ddp_model = DDP(model, device_ids=[rank])
    #if torch.cuda.device_count() > 1: 
    #    model = torch.nn.Distributed  DataParallel(model) # Data parallelism model wrapper
    #model.cuda(device) # send to gpu

    # define optimizer
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.01, weight_decay=5e-4)
    
    train_accuracies, test_accuracies = list(), list()
    start = time.time()
    for epoch in range(100):
        #for data in data_loader:
        input = data.to(rank)
        ddp_model.train()
        optimizer.zero_grad()
        out = ddp_model(input) # forward pass
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward() # backward pass
        optimizer.step() # update parameters

        train_acc = test(ddp_model, data) # training accuracy
        test_acc = test(ddp_model, data, train=False) # testing accuracy

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
              format(epoch, loss, train_acc, test_acc))
    end = time.time()
    print("Elapsed time: ", end-start)

    if plot:
        plt.plot(train_accuracies, label="Train accuracy")
        plt.plot(test_accuracies, label="Validation accuracy")
        plt.xlabel("# Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc='upper right')
        plt.savefig("AUC_praxidike97.png")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    #for name, param in model.named_parameters():
    #    print("name", name, param.device)

    # Train the model
    world_size = 2
    mp.spawn(train, args=(world_size,),nprocs=world_size, join=True)

'''
class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Step 1: Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Multiply with weights
        x = self.lin(x)

        # Step 3: Calculate the normalization
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4: Propagate the embeddings to the next layer
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x,
                              norm=norm)

    def message(self, x_j, norm):
        # Normalize node features.
        return norm.view(-1, 1) * x_j
'''