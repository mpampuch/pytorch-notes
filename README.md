# PyTorch Notes

Some tips / things of notes for myself while I'm learning PyTorch

## What is PyTorch?

PyTorch is an open-source machine learning library widely used in academia and industry for building deep learning models. It offers dynamic computation graphs that provide flexibility and ease in debugging, making it especially suited for research and complex model development. PyTorch accelerates the training of models through efficient computaitons. It seamlessly integrates with GPU for enhanced computational speed, making it ideal for handling large-scale data in AI applications. It is particularly useful in fields like genomics, where understanding patterns in large volumes of data can lead to breakthroughs in understanding complex natural phenomena.

![A PyTorch Workflow](what-is-pytorch.png)

## Running PyTorch

## Converting between Numpy arrays and PyTorch tensors

### PyTorch and Numpy have different default data types

### Numpy arrays cannot be on GPU, use `Tensor.cpu()` to convert PyTorch data on GPU to Numpy

## Random Seeds 
(need to be set twice) (IMG_4270.PNG)

## Device agnostic code

## `nn.Module` and the `forward` method
(IMG_4323.PNG)

## `model.parameters` vs `model.state_dict`

## `model.train()`, `model.eval()`, and `torch.inference_mode()` 
