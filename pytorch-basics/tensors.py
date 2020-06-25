import torch

# 5x3 uninitialised tensor
x = torch.empty(5, 3)
print(x)

# 5x3 randomly initialised  tensor
x = torch.randn(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.float32)
print(x)
print("z size", x.size())