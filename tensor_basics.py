import torch
import numpy as np

def tensor_examples():
    t1 = torch.tensor([[1,2], [3,4]])
    print('torch tensor from list: \n', t1) #torch tensor from given list

    arr = np.array([[5,6], [7,8]])
    t2 = torch.from_numpy(arr) # converts np array to pytorch tensor. 
    print('tensor from numpy: \n', t2) #torch tensor from np array

    print('zeros:', torch.zeros(2,3))
    print('random:', torch.rand(2,3))
    print('ones:', torch.ones(2,3)) #types of tensors

    print('device', t1.device) #will give cpu for me
    print('dtype', t1.dtype) # integer type


if __name__ == "__main__":
    tensor_examples() #safe entry point only run def fxn if main is run


## to save this file we write 
## python-m tensor_basics.py > tensor_results.txt