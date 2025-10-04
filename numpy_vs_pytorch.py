import numpy as np
import torch

np_array = np.array([[1,2], [3,4]])

tensor = torch.tensor([[4,12], [34,21]])

t1 = torch.from_numpy(np_array)

convertToNumpy = tensor.numpy()

print(t1)
print(convertToNumpy)

print(t1+1)

print(convertToNumpy * 2)


if torch.cuda.is_available():
    tensor_gpu = tensor.to('cuda')
    print("tensor on gpu: ", tensor)
else:
    print("cuda not available")

print(t1.dtype)
print(t1.device)
print(t1.shape)

print(convertToNumpy.dtype)
print(convertToNumpy.device)
print(convertToNumpy.shape)

## tensor and arrays are the same, its just that pytorch tensors can be put on gpus
