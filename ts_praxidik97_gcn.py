#!/opt/software/Python/3.6.4-foss-2018a/bin/python
"""
Convert PyTorch model to TorchScript

Source:  https://github.com/praxidike97/GraphNeuralNet/blob/master/main.py

To run on command line:
python praxidike97_gcn.py 
"""

import torch
import torchvision
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import networkx as nx
from torch_geometric.datasets import Planetoid
import numpy as np
import os
import time
import matplotlib.pyplot as plt



class GCNConv(MessagePassing): #
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  #,  "Add" aggregation
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Step 1: Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Multiply with weights
        x = self.lin(x)

        # Step 3: Calculate the normalization
        deg = degree(edge_index[0], x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]

        # Step 4: Propagate the embeddings to the next layer
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x,
                              norm=norm)

    def message(self, x_j, norm):
        # Normalize node features.
        return norm.view(-1, 1) * x_j

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1433, 16)
        self.conv2 = GCNConv(16, 7)

    def forward(self, x, edge_index):
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
    pred = model(data.x, data.edge_index).max(dim=1)[1]

    if train:
        correct += pred[data.train_mask].eq(data.y[data.train_mask]).sum().item()
        return correct / (len(data.y[data.train_mask]))
    else:
        correct += pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        return correct / (len(data.y[data.test_mask]))

def train(data, rank='cuda:1'): #world_size
    torch.cuda.empty_cache()
    print("GPUs", torch.cuda.device_count())
    print("GLOO", os.environ.get('GLOO_SOCKET_IFNAME'))
    
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print(t, r, a, f)

    # create local model
    model = Net().to(rank)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    train_accuracies, test_accuracies = list(), list()
    start = time.time()
    for epoch in range(100):
        input = data.to(rank)
        model.train()
        optimizer.zero_grad()
        out = model(input.x, input.edge_index) # forward pass
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward() # backward pass
        optimizer.step() # update parameters

        train_acc = test(model, data) # training accuracy
        test_acc = test(model, data, train=False) # testing accuracy

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
    
    """ Convert model to TorchScript via tracing """
    # dataset
    plot=False
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    x, edge_index = data.x, data.edge_index

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    mod = Net()
    mod.eval()
    traced_script_module = torch.jit.trace(mod(x, edge_index))

    # See TorchScript code
    print(traced_script_module.code)

    # Save the traced TorchScript model to a file
    traced_script_module.save("traced_praxidike97_gcn.pt")
    """ End """

    # Train the model
    train(data, rank='cuda:1')
