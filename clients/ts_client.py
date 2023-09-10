# This file is modified from Jordan's code

import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

from clients.fl_client import FLClient

class TSClient(FLClient):
  def train(self, net, optimizer, x, y, num_epoch=4, batch_size=128):
    print_every = -1

    for n in range(num_epoch):
      nts, ts1, ts2, ts3, ts4 = x
      permutation = torch.randperm(nts.size()[0])
      for i in range(0, nts.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        x_mini = (nts[indices], ts1[indices], ts2[indices], ts3[indices], ts4[indices])
        y_mini = y[indices]
        y_pred = net(x_mini)
        loss = nn.MSELoss()(y_pred, y_mini)
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        if print_every != -1 and (i / batch_size) % print_every == 0:
          print(f'Epoch: {n + 1}, Iteration: {round(i / batch_size)}, Loss: {loss.sum()}')
      if print_every == -1:
        print(f'Epoch: {n + 1}, Loss: {loss.sum()}')

  def test(self, net, x, y):
    with torch.no_grad():
      y_pred = net(x)
      return mean_squared_error(y.detach().numpy(), y_pred.detach().numpy()).item()