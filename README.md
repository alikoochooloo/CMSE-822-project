# CMSE-822-project

# Title: Machine Learning Parallelization of Genomic Prediction Models 
## Group members: Ali Saffary and Kenia Segura Abá

# Abstract:

Determining how genotype is connected to phenotype is a grand challenge in biology. Due to technological innovations, a large amount of biological data (i.e., multi-omic data) has been generated for many organisms, particularly plants. Thus, a crucial question is how this data can be leveraged to understand the genotype-to-phenotype relationship. However, the sheer volume and size of these multi-omics data pose serious analytical challenges. Therefore, our goal is to utilize machine learning tools to predict traits from genomic data, a type of multi-omic data, from the model plant Arabidopsis thaliana. This process is also known as genomic prediction. Using existing genomic data from Arabidopsis, we will build a Graph Convolutional Neural Network (GCN) for a regression task using the PyTorch Python package to predict flowering time. We will implement data parallelism in PyTorch using the DataParallel tool, which splits the input data that is fed into the model across multiple GPUs or CPUs. An alternative approach we will explore is to use multiple CPU threads for inter-op parallelism, where one or more threads execute the model’s operations on the given inputs, using TorchScript.

# Parallelization Strategies:

_Data Parallelism_
The genomic prediction model we are using has 3 graph convolutional layers with a ReLU activation layer following each of them. None of the layers are easily parallelizable as it would require us to fully redesign each layer to apply threading to the weight multiplications. However Pytorch has a tool available called DataParallel that can be used as follows:

	#model initialization
	model = torch.nn.DataParallel(model, device_ids)
	#call the model

DataParallel takes in a model and applies parallelism by replicating the given model and splitting the input training data, with a specific batch size, across multiple devices (GPUs & CPUs). When passing a batch of inputs (equal to or more than the number of cores available) the model will segment the batch and pass approximately the same number of samples to each core. We will build and test the GCN model in the HPCC and use a different number of device cores to benchmark how the training time will improve and if the model performance is affected.

_Inter-op parallelism_
Alternatively the model can be modified to include asynchronous sections that can run concurrently and be joined together later for the output layer. We will use TorchScript, which is an intermediate representation of PyTorch models to allow them to be run in a high-performance environment such as C++ and compiled using the PyTorch JIT compiler. This tool allows us to overcome Python’s GIL and improve model runtime. 

In which case we can use a forking tool available through TorchScript called torch.jit.fork. This operator can split several operations of the model to multiple threads and create a future object that can be used to call for a thread join.

To test this tool we would simply produce two different scripts which use the same model architecture, but one applies threading to the asynchronous section of the model. This experiment will result in a training speedup.

# Tools Used: 

Python, PyTorch, TorchScript, JIT, torch.nn.DataParallel, torch.jit.fork, HPCC, CUDA

# Benchmarking and Optimization:

Our reference to benchmark against would be the time it takes to run our machine learning model (before any parallelizing modifications) on the HPCC with 1 processing core available (GPU or CPU) on that session.

Success in our project would be defined by the level of improvement in training efficiency of the machine learning model through the implementation of Data Parallelism or Inter-op Parallelism. When PyTorch runs a sequence of operations on the specified device, it calls a subprogram (i.e., kernel) for each operation. Thus, each kernel reads, performs the calculation, and writes from the device memory, which increases latency. By parallelizing the input data or the operations, we expect that the model training time would not be decreased linearly as the amount of overhead is rather large, but there will be improvement in the training time since the read, calculation, and write operations will be done at once. 

# Dataset

The dataset we are using was previously published by [Alonso-Blanco *et al.* 2016](https://doi.org/10.1016/j.cell.2016.05.063). The original dataset contains a full VCF genomic variant file that combines all 1,135 *Arabidopsis thaliana* accessions, however we will be using a subset of 383 accessions. The data is freely available at http://1001genomes.org/data/GMI-MPI/releases/v3.1.

# References:
CPU threading and TorchScript inference¶. CPU threading and TorchScript inference - PyTorch 1.10.0 documentation. (n.d.). Retrieved October 29, 2021, from https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html. 

Optional: Data parallelism¶. Optional: Data Parallelism - PyTorch Tutorials 1.10.0+cu102 documentation. (n.d.). Retrieved October 29, 2021, from https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html. 

Stevens, E., Antiga, L., & Viehmann, T. (2020). Deep learning with PyTorch. Manning Publications.
