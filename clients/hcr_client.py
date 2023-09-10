# This file is modified from Jordan's code

import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from clients.fl_client import FLClient
from flwr.common.logger import log
from logging import DEBUG, INFO

class HCRClient(FLClient):
  def train(self, net, optimizer, x, y, num_epoch=64, batch_size=128):

    log(INFO, 'fit num_rounds')

    print_every = -1

    for n in range(num_epoch):
      permutation = torch.randperm(x.size()[0])
      for i in range(0, x.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        x_mini, y_mini = x[indices], y[indices]
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
    y_pred = net(x)
    test_mse(net, x, y)

    return mean_squared_error(y.detach().numpy(), y_pred.detach().numpy()).item()


