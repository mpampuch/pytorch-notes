# PyTorch Notes

Some tips / things of notes for myself while I'm learning PyTorch

## What is PyTorch?

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
