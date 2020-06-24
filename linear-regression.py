import torch as t
import torch.nn as nn
import torch.optim as optim

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)
        self.optim = optim.SGD(self.parameters(), lr=0.01)

    def forward(self, input):
        return self.linear(input)

x = t.tensor([[1.0], [2.0], [3.0], [4.0]])
y = t.tensor([[2.0], [4.0], [6.0], [8.0]])

model = LinearRegression()

for _ in range(100):
    y_pred = model(x)
    loss = nn.MSELoss()
    loss = loss(y_pred, y)
    model.optim.zero_grad()
    loss.backward()
    model.optim.step()
    print(loss)