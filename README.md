# CMSE-822-project

# Title: Machine Learning Parallelization of Genomic Prediction Models 
## Group members: Ali Saffary and Kenia Segura Abá

# Abstract:

Determining how genotype is connected to phenotype is a grand challenge in biology. Due to technological innovations, a large amount of biological data (i.e., multi-omic data) has been generated for many organisms, particularly plants. Thus, a crucial question is how this data can be leveraged to understand the genotype-to-phenotype relationship. However, the sheer volume and size of these multi-omics data pose serious analytical challenges. Therefore, our goal is to utilize machine learning tools to predict traits from genomic data, a type of multi-omic data, from the model plant Arabidopsis thaliana. This process is also known as genomic prediction. Using existing genomic data from Arabidopsis, we will build a Graph Convolutional Neural Network (GCN) for a regression task using the PyTorch Python package to predict flowering time. We will implement data parallelism in PyTorch using the DataParallel tool, which splits the input data that is fed into the model across multiple GPUs or CPUs. An alternative approach we will explore is to use multiple CPU threads for inter-op parallelism, where one or more threads execute the model’s operations on the given inputs, using TorchScript.

# Parallelization Strategies:

_Data Parallelism_

The genomic prediction model we are using has 2 graph convolutional layers with a ReLU activation layer following each of them. None of the layers are easily parallelizable as it would require us to fully redesign each layer to apply threading to the weight multiplications. However Pytorch has a tool available called DataParallel that can be used as follows:

```
#model initialization
model = torch.nn.DataParallel(model, device_ids)
#call the model
```

DataParallel takes in a model and applies parallelism by replicating the given model and splitting the input training data, with a specific batch size, across multiple devices (GPUs & CPUs). When passing a batch of inputs (equal to or more than the number of cores available) the model will segment the batch and pass approximately the same number of samples to each core. We will build and test the GCN model in the HPCC and use a different number of device cores to benchmark how the training time will improve and if the model performance is affected.

_Inter-op parallelism & CPU Threading_

Alternatively the model can be modified to include asynchronous sections that can run concurrently and be joined together later for the output layer. We will use TorchScript, which is an intermediate representation of PyTorch models to allow them to be run in a high-performance environment such as C++ and compiled using the PyTorch JIT compiler. This tool allows us to overcome Python’s GIL and improve model runtime. 

In which case we can use a forking tool available through TorchScript called torch.jit.fork. This operator can split several operations of the model to multiple threads and create a future object that can be used to call for a thread join.

To test this tool we would simply produce two different scripts which use the same model architecture, but one applies threading to the asynchronous section of the model. This experiment will result in a training speedup.

# Tools Used: 

Python, PyTorch, TorchScript, JIT, torch.nn.DataParallel, torch.jit.fork, HPCC, CUDA

# Benchmarking and Optimization:

Our reference to benchmark against would be the time it takes to run our machine learning model (before any parallelizing modifications) on the HPCC with 1 processing core available (GPU or CPU) on that session.

Success in our project would be defined by the level of improvement in training efficiency of the machine learning model through the implementation of Data Parallelism or Inter-op Parallelism. When PyTorch runs a sequence of operations on the specified device, it calls a subprogram (i.e., kernel) for each operation. Thus, each kernel reads, performs the calculation, and writes from the device memory, which increases latency. By parallelizing the input data or the operations, we expect that the model training time would not be decreased linearly as the amount of overhead is rather large, but there will be improvement in the training time since the read, calculation, and write operations will be done at once. 

# Dataset:

The dataset we are using to predict flowering time using the GCN was previously published by [Alonso-Blanco *et al.* 2016](https://doi.org/10.1016/j.cell.2016.05.063). The original dataset contains a full VCF genomic variant file that combines all 1,135 *Arabidopsis thaliana* accessions, however we will be using a subset of 383 accessions. The data is freely available at http://1001genomes.org/data/GMI-MPI/releases/v3.1.

A secondary dataset we are using to develop the GCN is the PyTorch Geometric [Planetoid](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.Planetoid) dataset, which has the correct data structures and format to run on PyTorch models

# Description of code:
- praxidike97_gcn.py: GCN for classification using Distributed Data Parallel
- praxidike97_gcn_dp.py: GCN for classification using DataParallel
- praxidike97_gcn_original.py: GCN for classification NO PARALLELIZATION
- praxidike97_gcn_regression.py: GCN for regression (genotype data) NO PARALLELIZATION >> this code only worked on planetoid data
- ts_praxidike97_gcn_original.py: Code to trace GCN for classification into TorchScript
- traced_praxidike97_gcn_original.pt: Traced TorchScript model of GCN for classification
- example_ddp.py: Distributed Data Parallel example
- example_dp.py: DataParallel example

# Challenges in Troubleshooting the GCN:
__PyTorch DataParallel:__
Without implementing the data parallelism strategy, the GCN was trained smoothly, however, the DataParallel() tool would produce the following error:
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:1! (when checking arugment for argument mat1 in method wrapper_addmm)
```
We followed this [example](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html), since when executed, we obtained the expected output as the post describes and it appeared simple to implement, but this was not the case with our GCN. We looked to this [PyTorch discussion post](https://discuss.pytorch.org/t/tensors-on-different-devices-when-using-dataparallel-expected-all-tensors-to-be-on-the-same-device/127952) with the intention of understanding the runtime error but the problem described was about initiliazing lists the normal way and not the PyTorch way (using nn.ModuleList). We did not initialize lists in our GCN so this post was not helpful and it was also the only post that we found discussing this error in conjunction with DataParallel(). Thus, we were unable to solve this error.

__PyTorch DistributedDataParallel:__
The next thing we tried was to implement Distributed Data Parallelism, which spawns multiple processes and creates a single instance of the model per process. DistributedDataParallel() was implemented as described in this [example](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html):
```
from torch.nn.parallel import DistributedDataParallel as DDP

# create default process group
dist.init_process_group(backend="nccl")

# create local model
model = Net(dataset).to(rank) # rank is device ID which is automatically set by torch.distributed.run

# construct distributed data parallel model
ddp_model = DDP(model, device_ids=[rank])

# In the main() function:
import torch.multiprocessing as mp
world_size = 2 # gpus
mp.spawn(train, args=(world_size,),nprocs=world_size, join=True) # function to train the model

# on command line use:
python -m torch.distributed.run praxidike97_gcn.py 
```
PyTorch also has multiple backends that can be used to run DistributedDataParallel(), GLOO and NCCL, and both support CPU and GPU tensors. We chose to use both since the NCCL backend directly supports distributed data parallelism on GPU and the GLOO backend supports CPU. We were not able to go far on GLOO because of a connection error we came across:
```
-- Process 1 terminated with the following error:
Traceback (most recent call last):
  File "/mnt/home/seguraab/miniconda3/envs/torch-env/lib/python3.9/site-packages/torch/multiprocessing/spawn.py", line 59, in _wrap
    fn(i, *args)
  File "/mnt/ufs18/home-056/seguraab/Shiu_Lab/Collabs/Multi_Omic/CMSE-822-project/praxidike97_gcn.py", line 113, in train
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
  File "/mnt/home/seguraab/miniconda3/envs/torch-env/lib/python3.9/site-packages/torch/distributed/distributed_c10d.py", line 523, in init_process_group
    default_pg = _new_process_group_helper(
  File "/mnt/home/seguraab/miniconda3/envs/torch-env/lib/python3.9/site-packages/torch/distributed/distributed_c10d.py", line 616, in _new_process_group_helper
    pg = ProcessGroupGloo(
RuntimeError: [/opt/conda/conda-bld/pytorch_1623448255797/work/third_party/gloo/gloo/transport/tcp/pair.cc:799] connect [192.168.0.222]:2226: Connection refused
```
We were unable to find a solution to this error since it appeared that conda was refusing to connect. The configurations for the GLOO backend were these:
```
[INFO] 2021-12-03 16:10:13,734 api: Starting elastic_operator with launch configs:
  entrypoint       : praxidike97_gcn.py
  min_nodes        : 1
  max_nodes        : 1
  nproc_per_node   : 4
  run_id           : none
  rdzv_backend     : static
  rdzv_endpoint    : 127.0.0.1:29500
  rdzv_configs     : {'rank': 0, 'timeout': 900}
  max_restarts     : 3
  monitor_interval : 5
  log_dir          : None
  metrics_cfg      : {}
```
At this point, we were completely stuck and could not find a solution to our problem, so we moved on to try the NCCL backend. The configurations for the NCCL backend were the same as for the GLOO, but this time we did not have an error. Instead, each of the processes hanged and ran for hours without output. We ran the GCN using the command `NCCL_ASYNC_ERROR_HANDLING=1 python -m torch.distributed.run praxidike97_gcn.py` because we thought that the asynchronous processes might have been blocked, but a CUDA_LAUNCH_BLOCKING error was printed, which meant that the CUDA device was blocking the asynchronous processes:
```
RuntimeError: CUDA error: out of memory
CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
```
The `NCCL_ASYNC_ERROR_HANDLING=1` flag was set to set the duration time after which processes will be aborted asynchronously. According to the PyTorch documentation on distributed process groups ([here](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group)), aborting processes is necessary since CUDA execution is also asynchronous and could lead to asyncronous NCCL operations to run on corrupted data from the CUDA operations. Thus, we re-ran the GCN with the command `NCCL_ASYNC_ERROR_HANDLING=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.run praxidike97_gcn.py` and set the available CUDA GPU devices to IDs 1-4. There was no error this time, but the processes hanged again.

__TorchScript:__
1. I “traced” the model to convert it to torchscript but there isn’t a straightforward way to parallelize in C++ and I can’t find a good example

__Intra-op parallelism with OpenMP:__
PyTorch supports intra-op parallelism (parallelization within an operation) with [OpenMP](https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html) to parallelize for loops. In our code, we train the GCN for 100 epochs and we set the environment variable OMP_NUM_THREADS to variables ranging between 1-7 (see the table below), for example, `OMP_NUM_THREADS=2 taskset -c 0 python praxidike97_gcn_original.py`. The `taskset -c` flag sets the device (CUDA device 0).

|Threads|Time (s)|-|# of Devices|Threads|Time (s)|
|-------|--------|-|------------|-------|--------|
|1	|1.06712|-|2  |1  |3.31052|
|2	|0.97997|-|2  |2  |__0.80018__|
|3	|0.85818|-|2  |3  |0.87675|
|4	|__0.84929__|-|2  |4  |0.92798|
|5	|1.26594|-|4  |1  |1.03776|
|6	|1.03312|-|4  |2  |__0.98101__|
|7	|3.73669|-|4  |3  |1.06247|
|   |       |-|4  |4  |1.15296|


When we did not set the OMP_NUM_THREADS variable, model training took 1.34716 seconds. Overall, we see an optimal training speed up using 3-4 threads on one device. If too many threads are used, oversubscription causes an increase in training time due to conflicts over access to shared data in the device memory. 

# References:
Alonso-Blanco, C., Andrade, J., Becker, C., Bemm, F., Bergelson, J., Borgwardt, K. M., ... & Zhou, X. (2016). 1,135 genomes reveal the global pattern of polymorphism in Arabidopsis thaliana. Cell, 166(2), 481-491.

CPU threading and TorchScript inference¶. CPU threading and TorchScript inference - PyTorch 1.10.0 documentation. (n.d.). Retrieved October 29, 2021, from https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html. 

Optional: Data parallelism¶. Optional: Data Parallelism - PyTorch Tutorials 1.10.0+cu102 documentation. (n.d.). Retrieved October 29, 2021, from https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html. 

Stevens, E., Antiga, L., & Viehmann, T. (2020). Deep learning with PyTorch. Manning Publications.
