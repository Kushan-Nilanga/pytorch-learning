import torch

x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
y = torch.tensor([2.0, 4.0, 6.0], dtype=torch.float32)

## Normal Operations
# addition
add = x + y
print(add)

# substraction
sub = y - x
print(sub)

# multiplication
mul = y * x
print(mul)

# division
div = y / x
print(div)

## Inplace in Operations
# addition
print(y)
y.add_(x)
print(y)