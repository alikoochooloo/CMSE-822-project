# https://github.com/praxidike97/GraphNeuralNet/blob/master/main.py
# To run on command line
# CUDA_VISIBLE_DEVICES=0,1 CUDA_LAUNCH_BLOCKING=1 python multi-omics/GCN/praxidike97_gcn.py 
from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.utils.data as data
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
import torch.multiprocessing as mp
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time

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

class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
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


def test(data, train=True):
    model.eval()

    correct = 0
    pred = model(data).max(dim=1)[1]

    if train:
        correct += pred[data.train_mask].eq(data.y[data.train_mask]).sum().item()
        return correct / (len(data.y[data.train_mask]))
    else:
        correct += pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        return correct / (len(data.y[data.test_mask]))


def train(args):
    plot=False
    dataset = Planetoid(root='/tmp/Cora', name='Cora')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # check device
    print(device)
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print(t, r, a, f)

    model = Net(dataset) # create GCN model
    print("GPUs", torch.cuda.device_count())
    if torch.cuda.device_count() > 1: 
        model = torch.nn.DataParallel(model) # Data parallelism model wrapper
    model.cuda(device) # send to gpu

    data_loader = DataLoader(dataset=dataset, batch_size = 30, shuffle=True, num_workers=0, pin_memory=True)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    train_accuracies, test_accuracies = list(), list()
    start = time.time()
    for epoch in range(100):
        for data in data_loader:
            input = data.cuda(device)
            model.train()
            optimizer.zero_grad()
            out = model(input)
            print("Outside: input size", input.size(), "output_size", out.size())
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            train_acc = test(data)
            test_acc = test(data, train=False)

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
        plt.show()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    #Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])
    #plot_dataset(dataset)

    

    
    #for name, param in model.named_parameters():
    #    print("name", name, param.device)

    # Train the model
    mp.spawn(train)
    
# To try: slurm script
'''
(torch-env) [seguraab@dev-amd20-v100 Multi_Omic]$ CUDA_VISIBLE_DEVICES=0,1 CUDA_LAUNCH_BLOCKING=1 python multi-omics/GCN/praxidike97_gcn.py 
cuda
34089730048 0 0 0
GPUs 2
/mnt/home/seguraab/miniconda3/envs/torch-env/lib/python3.9/site-packages/torch_geometric/deprecation.py:13: UserWarning: 'data.DataLoader' is deprecated, use 'loader.DataLoader' instead
  warnings.warn(out)
Traceback (most recent call last):
  File "/mnt/ufs18/home-056/seguraab/Shiu_Lab/Collabs/Multi_Omic/multi-omics/GCN/praxidike97_gcn.py", line 157, in <module>
    mp.spawn(train)
  File "/mnt/home/seguraab/miniconda3/envs/torch-env/lib/python3.9/site-packages/torch/multiprocessing/spawn.py", line 230, in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method='spawn')
  File "/mnt/home/seguraab/miniconda3/envs/torch-env/lib/python3.9/site-packages/torch/multiprocessing/spawn.py", line 188, in start_processes
    while not context.join():
  File "/mnt/home/seguraab/miniconda3/envs/torch-env/lib/python3.9/site-packages/torch/multiprocessing/spawn.py", line 150, in join
    raise ProcessRaisedException(msg, error_index, failed_process.pid)
torch.multiprocessing.spawn.ProcessRaisedException: 

-- Process 0 terminated with the following error:
Traceback (most recent call last):
  File "/mnt/home/seguraab/miniconda3/envs/torch-env/lib/python3.9/site-packages/torch/multiprocessing/spawn.py", line 59, in _wrap
    fn(i, *args)
  File "/mnt/ufs18/home-056/seguraab/Shiu_Lab/Collabs/Multi_Omic/multi-omics/GCN/praxidike97_gcn.py", line 120, in train
    out = model(input)
  File "/mnt/home/seguraab/miniconda3/envs/torch-env/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1051, in _call_impl
    return forward_call(*input, **kwargs)
  File "/mnt/home/seguraab/miniconda3/envs/torch-env/lib/python3.9/site-packages/torch/nn/parallel/data_parallel.py", line 168, in forward
    outputs = self.parallel_apply(replicas, inputs, kwargs)
  File "/mnt/home/seguraab/miniconda3/envs/torch-env/lib/python3.9/site-packages/torch/nn/parallel/data_parallel.py", line 178, in parallel_apply
    return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])
  File "/mnt/home/seguraab/miniconda3/envs/torch-env/lib/python3.9/site-packages/torch/nn/parallel/parallel_apply.py", line 86, in parallel_apply
    output.reraise()
  File "/mnt/home/seguraab/miniconda3/envs/torch-env/lib/python3.9/site-packages/torch/_utils.py", line 425, in reraise
    raise self.exc_type(msg)
RuntimeError: Caught RuntimeError in replica 1 on device 1.
Original Traceback (most recent call last):
  File "/mnt/home/seguraab/miniconda3/envs/torch-env/lib/python3.9/site-packages/torch/nn/parallel/parallel_apply.py", line 61, in _worker
    output = module(*input, **kwargs)
  File "/mnt/home/seguraab/miniconda3/envs/torch-env/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1051, in _call_impl
    return forward_call(*input, **kwargs)
  File "/mnt/ufs18/home-056/seguraab/Shiu_Lab/Collabs/Multi_Omic/multi-omics/GCN/praxidike97_gcn.py", line 52, in forward
    x = self.conv1(x, edge_index)
  File "/mnt/home/seguraab/miniconda3/envs/torch-env/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1051, in _call_impl
    return forward_call(*input, **kwargs)
  File "/mnt/home/seguraab/miniconda3/envs/torch-env/lib/python3.9/site-packages/torch_geometric/nn/conv/gcn_conv.py", line 181, in forward
    x = self.lin(x)
  File "/mnt/home/seguraab/miniconda3/envs/torch-env/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1051, in _call_impl
    return forward_call(*input, **kwargs)
  File "/mnt/home/seguraab/miniconda3/envs/torch-env/lib/python3.9/site-packages/torch_geometric/nn/dense/linear.py", line 102, in forward
    return F.linear(x, self.weight, self.bias)
  File "/mnt/home/seguraab/miniconda3/envs/torch-env/lib/python3.9/site-packages/torch/nn/functional.py", line 1847, in linear
    return torch._C._nn.linear(input, weight, bias)
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:1! (when checking arugment for argument mat2 in method wrapper_mm)
'''