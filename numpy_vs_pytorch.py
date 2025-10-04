import numpy as np
import torch

np_array = np.array([[1,2], [3,4]])

tensor = torch.tensor([[4,12], [34,21]])

t1 = torch.from_numpy(np_array)

convertToNumpy = tensor.numpy()

print(t1)
print(convertToNumpy)


if torch.cuda.is_available():
    tensor_gpu = tensor.to('cuda')
    print("tensor on gpu: ", tensor)
else:
    print("cuda not available")



## tensor and arrays are the same, its just that pytorch tensors can be put on gpus
