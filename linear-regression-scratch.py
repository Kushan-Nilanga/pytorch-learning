import torch
import torch.nn as nn
import torch.optim as optim

x = torch.tensor([1, 2, 3, 4])
w = torch.tensor(1.0, requires_grad=True)

y = torch.tensor([2, 4, 6, 8])

for _ in range(100):
    fx = x * w
    loss = (y - fx) ** 2
    loss = loss.mean()
    loss.backward()
    with torch.no_grad():
        w -= w.grad * 0.01 * loss
    w.grad.zero_()
    print(loss.item(), w.item())
    