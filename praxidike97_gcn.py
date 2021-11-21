#!/opt/software/Python/3.6.4-foss-2018a/bin/python
"""
 https://github.com/praxidike97/GraphNeuralNet/blob/master/main.py

To run on command line (no DDP):
python praxidike97_gcn.py

If CUDA errors try (no DDP):
    CUDA_VISIBLE_DEVICES=1,2 CUDA_LAUNCH_BLOCKING=1 python praxidike97_gcn.py 

With Distributed Parallel:
    python -m torch.distributed.run praxidike97_gcn.py
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run praxidike97_gcn.py
    python -m torch.distributed.launch --nproc_per_node=2 --nnode=2 --node_rank=0 praxidike97_gcn.py

With OpenMP on different GPUs (0,1,2,3) or with different number of threads: 
    # source: https://github.com/pytorch/pytorch/issues/3146
    GOMP_CPU_AFFINITY="0" OMP_NUM_THREADS=1 python praxidike97_gcn.py
    OMP_NUM_THREADS=1 taskset -c 0 python praxidike97_gcn.py

With MPI backend:
    RuntimeError: Distributed package doesn't have MPI built in. MPI is only 
    included if you build PyTorch from source on a host that has MPI installed.
    # source: https://pytorch.org/tutorials/intermediate/dist_tuto.html
    mpirun -n 2 python myscript.py

Check running processes: ps -elf | grep python
"""

from torch_geometric.datasets import Planetoid
import torch
import torchvision
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
    def __init__(self, dataset, dev0, dev1):
        super().__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.to(self.dev0)
        edge_index = edge_index.to(self.dev0)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = x.to(self.dev1)
        edge_index = edge_index.to(self.dev1)
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
    data = data.to('cuda:1')
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
    print("I am processor rank", rank)

    # dataset
    plot=False
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    data_loader = DataLoader(dataset=dataset, batch_size = 30, shuffle=True, num_workers=0, pin_memory=False)

    # create default process group
    dist.init_process_group(backend="mpi", rank=rank, world_size=world_size)

    # create local model
    dev0 = (rank * 2) % world_size
    dev1 = (rank * 2 + 1) % world_size
    print('dev0', dev0, 'dev1', dev1)
    model = Net(dataset, dev0, dev1).to(rank)

    # construct distributed data parallel model
    ddp_model = DDP(model, device_ids=[rank])#, output_device=dev1)

    # define optimizer
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.01, weight_decay=5e-4)
    
    train_accuracies, test_accuracies = list(), list()
    start = time.time()
    for epoch in range(100):
        for data in data_loader:
            input = data.to(rank)
            ddp_model.train()
            optimizer.zero_grad()
            out = ddp_model(input) # forward pass
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward() # backward pass
            optimizer.step() # update parameters

            train_acc = test(ddp_model, data.to(dev1)) # training accuracy
            test_acc = test(ddp_model, data.to(dev1), train=False) # testing accuracy

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

    dist.destroy_process_group() # clean up

if __name__ == "__main__":
    torch.cuda.empty_cache()

    #for name, param in model.named_parameters():
    #    print("name", name, param.device)

    # Train the model
    world_size = 2
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    #train(rank)

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