import torch
import numpy as np



data = [[1,2],[3,4]]
x_data = torch.tensor(data)



np_array = np.array(data)
x_np = torch.from_numpy(np_array)



x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data,dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")



x_data


shape = (2,3,)



shape


rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)



rand_tensor


ones_tensor


zeros_tensor


tensor = torch.rand(3,4)



print(tensor.dtype)



if torch.cuda.is_available():
    tensor = torch.to('cuda')



tensor = torch.ones(4,4)
tensor[:,1] = 0
print(tensor)


t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)



print(f"tensor.mul(tensor) : \n {tensor.mul(tensor)} \n")
    
print(f"tensor*tensor : \n {tensor * tensor} \n")




print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.t)} \n")
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")



print(tensor, "\n")
tensor.add_(5)
print(tensor)


t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")


t.add_(1)
print(f"t: {t} ")
print(f"n: {n}")


n


t


n = np.ones(5)
t = torch.from_numpy(n)


np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")



