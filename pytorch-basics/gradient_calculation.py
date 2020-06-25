import torch

x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
w = torch.tensor([1], dtype=torch.float32, requires_grad=True)
b = torch.tensor([1], dtype=torch.float32, requires_grad=True)

y_pred = torch.tensor([3, 6, 9, 12, 15], dtype=torch.float32)

for i in range(10000):
    y = w * x + b
    loss = ((y - y_pred) ** 2) / y.size()[0]
    loss = loss.mean()
    loss.backward()
    with torch.no_grad():
        w -= w * w.grad * 0.05
        b -= b*b.grad * 0.05
    w.grad.zero_()
    b.grad.zero_()
    if i % 10 == 0:
        print(loss.item(), w.item(), b.item())
