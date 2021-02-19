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
print(tensor.shape)
print(tensor.device)


tensor



