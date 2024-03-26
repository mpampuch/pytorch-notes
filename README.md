# PyTorch Notes

Some tips / things of notes for myself while I'm learning PyTorch

## What is PyTorch?

PyTorch is an open-source machine learning library widely used in academia and industry for building deep learning models. It offers dynamic computation graphs that provide flexibility and ease in debugging, making it especially suited for research and complex model development. PyTorch accelerates the training of models through efficient computaitons. It seamlessly integrates with GPU for enhanced computational speed, making it ideal for handling large-scale data in AI applications. It is particularly useful in fields like genomics, where understanding patterns in large volumes of data can lead to breakthroughs in understanding complex natural phenomena.

![A PyTorch Workflow](what-is-pytorch.png)

## Running PyTorch

The best place to get started running PyTorch code is with [Google Colab](https://colab.research.google.com/). It is a cloud-based Jupyter notebook environment provided by Google that allows users to write and execute Python code collaboratively in real-time. It comes pre-installed with PyTorch and a bunch of other useful libraries for data science and machine learning and it offers free access to GPUs and TPUs (this access is not guaranteed 24/7 but a lot of time you will be allocated a free device). 

## Tensors

Tensors are mathematical objects that generalize scalars, vectors, and matrices to higher dimensions. They are fundamental objects in linear algebra. 

Tensors are used extensively in various fields, including physics (especially in theories like general relativity and electromagnetism), engineering (for modeling stress and strain in materials), computer science (in machine learning, particularly in deep learning frameworks), and many other areas where multidimensional data needs to be manipulated and analyzed.

All data inputted and outputted out of models in PyTorch need to be represented as tensors.

### Converting between Numpy arrays and PyTorch tensors

Converting data between NumPy arrays and PyTorch tensors is a routine task. PyTorch tensors are similar to NumPy’s ndarrays, except that tensors can run on GPUs or other hardware accelerators. In fact, tensors and NumPy arrays can often share the same underlying memory, eliminating the need to copy data. There's 2 important difference that you have to remember when converting between the two objects.

1. PyTorch and Numpy have different default data types
    - NumPy arrays typically have a datatype of `float64` for floating-point numbers and `int64` for integers on most systems.
    - PyTorch tensors have a default datatype of `float32`
2. Numpy arrays cannot exist on a GPU, only a CPU, whereas PyTorch tensors can exist on both.

To convert NumPy arrays to PyTorch tensors, use the followings code and tools.

- Data in NumPy, want in PyTorch tensor -> `torch.from_numpy(ndarray)`
- PyTorch tensor, want in NumPy array -> `torch.Tensor.numpy()`
- For type coersion, `torch.from_numpy(ndarray).type(torch.float32)` 
- To convert PyTorch data on GPU to Numpy, use `Tensor.cpu()`


## Random Seeds 

Random seeds can be set before generating random data to ensure reproducibility, however setting manual seeds for random number generators in deep learning frameworks like PyTorch or TensorFlow is not generally recommended, because you want to make sure that the parameters your model learns are similar regardless of what random weights and biases they started with. However, if you do want to set random seeds, there are a few things you should know

You can set a random seed using 
```python
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
```

However be aware that random seeds aren't applied to every random generation in the code.

Example:

```python
# Let's make some random but reproducible tensors
import torch

# Set the random seed
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED) # SEED SET

# Generate random tensors
random_tensor_A = torch.rand(3,4)
random_tensor_B = torch.rand(3, 4)

# Print random tensors
print(random_tensor_A)
print(random_tensor_B)

# Check if random tensors are equal
print(random_tensor_A == random_tensor_B)
```

Outputs

```
tensor([[0.8823, 0.9150, 0.3829, 0.9593],
        [0.3904, 0.6009, 0.2566, 0.7936],
        [0.9408, 0.1332, 0.9346, 0.5936]])
tensor([[0.8694, 0.5677, 0.7411, 0.4294],
        [0.8854, 0.5739, 0.2666, 0.6274],
        [0.2696, 0.4414, 0.2969, 0.8317]])
tensor([[False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]])
```

To fix this issue, keep setting the same random seed before every random generation.

Example

```python
# Let's make some random but reproducible tensors
import torch

# Set the random seed
RANDOM_SEED = 42

# Generate random tensors
torch.manual_seed(RANDOM_SEED) # FIRST SEED SET
random_tensor_A = torch.rand(3,4)

torch.manual_seed(RANDOM_SEED) # SECOND SEED SET
random_tensor_B = torch.rand(3, 4)

# Print random tensors
print(random_tensor_A)
print(random_tensor_B)

# Check if random tensors are equal
print(random_tensor_A == random_tensor_B)
```

Outputs

```
tensor([[0.8823, 0.9150, 0.3829, 0.9593],
        [0.3904, 0.6009, 0.2566, 0.7936],
        [0.9408, 0.1332, 0.9346, 0.5936]])
tensor([[0.8823, 0.9150, 0.3829, 0.9593],
        [0.3904, 0.6009, 0.2566, 0.7936],
        [0.9408, 0.1332, 0.9346, 0.5936]])
tensor([[True, True, True, True],
        [True, True, True, True],
        [True, True, True, True]])
```

Additionally, if you want to set the manual seed on a GPU, use

```python
torch.cuda.manual_seed(seed)
```

## Device agnostic code

One of the most common errors in deep learning code is encountering a device error with some data or object in code. For example, you could try to multiply a tensor on a CPU with a tensor on a GPU and these two tensors will not be able to *find each other*, resulting in an error. 

Good machine learning code should never run into this error because the code should be **device agnostic**. To set up device agnostic code, write this code near the top of your script

```python
# Set up device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
```

Then when you want to set the device attribute on piece of data or model, write 

```python
torch.Tensor.to(device)
```

This will ensure all data is on the same device, especially if a GPU is not available.

**Note:** If you are working with NumPy data, that data can only exist on a CPU. If a GPU is available and PyTorch data is sent to the GPU, you can still encounter device errors when working with these two types of data. Use `torch.Tensor.cpu()` on NumPy data to fix this.

## Creating a machine learning model in PyTorch
![Creating a machine learning model diagram](creating-a-model.png)

## `nn.Module` and the `forward` method
(IMG_4323.PNG)

## Loss functions
- BCEWithLogitsLoss() # Binary classification
    - Combines sigmoid activation function with BCELoss. Better than BCELoss because more numerically stable

## Saving a model 



## Organising PyTorch projects

Once you have a machine learning project that you want to save, it may be a good idea to organize it into modules. 

![Modules](modules.png)

You may want the model to be launched from a command line executable. 

![PyTorch from the command line](command-line-pytorch.png)

Notebooks are fantastic for iteratively exploring and running experiments quickly.

However, for larger scale projects you may find Python scripts more reproducible and easier to run.

There's arguments for both sides.

| |**Pros**|**Cons**|
|:----|:----|:----|
|**Notebooks**|Easy to experiment/get started|Versioning can be hard|
| |Easy to share (e.g. a link to a Google Colab notebook)|Hard to use only specific parts|
| |Very visual|Text and graphics can get in the way of code|
| | 
|**Python scripts**|Can package code together (saves rewriting similar code across different notebooks)|Experimenting isn't as visual (usually have to run the whole script rather than one cell)|
| |Can use git for versioning| |
| |Many open source projects use scripts| |
| |Larger projects can be run on cloud vendors (not as much support for notebooks)| |

In anycase, a good workflow is to start on Google Colab and move important code to scripts if you would like

![PyTorch Workflow](pytorch-workflow.png)

### Jupyter cell magic commands

Jupyter cell magic commands are special commands that can be used in Jupyter notebooks to perform various tasks or change the behavior of code cells. These commands are preceded by one or two percent signs (`%` or `%%`) and are often used to streamline common tasks, interact with the notebook environment, or access system-level functionalities.

A very useful magic command to help convert the code in the code cell into a script is the `%%writefile` command. It's used as follows:

```
%%writefile DIR_NAME/SCRIPT_NAME.py

# Your Python code goes here
```

Executing this code cell will write all the code in the code cell into a script in the provided path.


## `model.parameters` vs `model.state_dict`



## `model.train()`, `model.eval()`, and `torch.inference_mode()` 


## Multiples of 8

Multiples of 8 are often recommended for machine learning, particularly in neural network architectures, due to their compatibility with hardware optimizations, specifically related to parallel processing and memory alignment. You should strive to use as many multiples of 8 (i.e. 8, 16, 32, 64, 128, 256, etc.) for hyperparameters such as number of units in hidden layers, batch sizes, etc.

### Minibatches, multiples of 8, and not going higher than 32

Mini-batches are subsets of a dataset used during the training process of machine learning models, particularly in deep learning. Instead of feeding the entire dataset into the model at once, mini-batch training involves dividing the dataset into smaller batches and processing each batch sequentially during training. 

In that same vain as above, minibatches should also be set to multiples of 8, but also it is often recommended to use minibatches of around 32, as this strikes a balance between utilizing GPU memory efficiently and allowing for a reasonable training speed.

## SciKitLearn train/test split

## TorchMetrics

## `torch.utils.data.DataSet` and `torch.utils.data.DataLoader`

## `timeit`

`from timeit import default_timer as timer`

## tqdm

```python
from tqdm.auto import tqdm
for i in tqdm(range(10000)):
    ...
```

## Transfer learning 

Transfer learning is a machine learning technique where a model trained on one task is adapted or transferred to a related but different task. Instead of training a model from scratch, transfer learning leverages knowledge gained from solving one problem to help solve a different but related problem. This approach is particularly useful when working with limited data or computational resources and can significantly reduce training time and improve performance.

Often times machine learning architects will start a new data science project by using another trained model as a backbone and modifying it for their purpose 

### "Freezing" and "Unfreezing"

### Resources for finding pre-trained models 

|**Location**|**What's there?**|**Link(s)**|
|:----|:----|:----|
|**PyTorch domain libraries**|Each of the PyTorch domain libraries (`torchvision`, `torchtext`) come with pretrained models of some form. The models there work right within PyTorch.|[`torchvision.models`](https://pytorch.org/vision/stable/models.html), [`torchtext.models`](https://pytorch.org/text/main/models.html), [`torchaudio.models`](https://pytorch.org/audio/stable/models.html), [`torchrec.models`](https://pytorch.org/torchrec/torchrec.models.html)|
|**HuggingFace Hub**|A series of pretrained models on many different domains (vision, text, audio and more) from organizations around the world. There's plenty of different datasets too.|https://huggingface.co/models, https://huggingface.co/datasets|
|**`timm` (PyTorch Image Models) library**|Almost all of the latest and greatest computer vision models in PyTorch code as well as plenty of other helpful computer vision features.|https://github.com/rwightman/pytorch-image-models|
|**Paperswithcode**|A collection of the latest state-of-the-art machine learning papers with code implementations attached. You can also find benchmarks here of model performance on different tasks.|https://paperswithcode.com/|

## Experiment tracking

If you're running lots of different experiments, experiment tracking helps you figure out what works and what doesn't.


Why track experiments?
If you're only running a handful of models (like we've done so far), it might be okay just to track their results in print outs and a few dictionaries.

However, as the number of experiments you run starts to increase, this naive way of tracking could get out of hand. So if you keep experimenting with models you'll want a way to track them.

There are as many different ways to track machine learning experiments as there is experiments to run.

This table covers a few.

|**Method**|**Setup**|**Pros**|**Cons**|**Cost**|
|:----|:----|:----|:----|:----|
|Python dictionaries, CSV files, print outs|None|Easy to setup, runs in pure Python|Hard to keep track of large numbers of experiments|Free|
|[TensorBoard](https://www.tensorflow.org/tensorboard/get_started)|Minimal, install [`tensorboard`](https://pypi.org/project/tensorboard/)|Extensions built into PyTorch, widely recognized and used, easily scales.|User-experience not as nice as other options.|Free|
|[Weights & Biases Experiment Tracking](https://wandb.ai/site/experiment-tracking)|Minimal, install [`wandb`](https://docs.wandb.ai/quickstart), make an account|Incredible user experience, make experiments public, tracks almost anything.|Requires external resource outside of PyTorch.|Free for personal use|
|[MLFlow](https://mlflow.org/)|Minimal, install `mlflow` and starting tracking|Fully open-source MLOps lifecycle management, many integrations.|Little bit harder to setup a remote tracking server than other services.|Free|

### TensorBoard

TensorBoard is a part of the TensorFlow deep learning library and is an excellent way to visualize different parts of your model.

### Creating Tensorboard compatible files

We can use PyTorch's `torch.utils.tensorboard.SummaryWriter()` class to save various parts of our model's training progress to file.

By default, the `SummaryWriter()` class saves various information about our model to a file set by the `log_dir` parameter.

The default location for `log_dir` is under `runs/CURRENT_DATETIME_HOSTNAME`, where the `HOSTNAME` is the name of your computer. You can change where your experiments are tracked and customize the filename as you'd like.

The outputs of the `SummaryWriter()` are saved in TensorBoard format, which makes them compatible with the TensorBoard display.



