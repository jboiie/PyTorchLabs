import torch

def operations():
    x = torch.tensor([[1,2], [3,4]], dtype=torch.float32)
    y = torch.tensor([[5,6], [7,8]], dtype=torch.float32)

    print('add: ', x + y)
    print('mult element wise:' , x*y)
    print ('matrix mult: ', torch.matmul(x,y))
    print("transpose: ", x.T)
    x.add_(1)
    print("a after add_1:", x)



if __name__ == "__main__":
    operations()   