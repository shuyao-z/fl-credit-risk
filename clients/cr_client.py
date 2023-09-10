# This file is modified from Jordan's code
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error
from clients.fl_client import FLClient

class CRClient(FLClient):
  def train(self, net, optimizer, x, y, num_epoch=256, batch_size=16):
    def ordinal_criterion(predictions, targets):
      modified_target = torch.zeros_like(predictions)
      for i, target in enumerate(targets):
        modified_target[i, 0:int(target)+1] = 1

      return nn.MSELoss(reduction='none')(predictions, modified_target).sum(axis=1)

    print_every = -1

    for n in range(num_epoch):
      permutation = torch.randperm(x.size()[0])
      for i in range(0, x.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        x_mini, y_mini = x[indices], y[indices]
        y_pred = net(x_mini)
        loss = ordinal_criterion(y_pred, y_mini)
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        if print_every != -1 and (i / batch_size) % print_every == 0:
          print(f'Epoch: {n + 1}, Iteration: {round(i / batch_size)}, Loss: {loss.sum()}')
      if print_every == -1:
        print(f'Epoch: {n + 1}, Loss: {loss.sum()}')

  def test(self, net, x, y):
    def prediction2label(pred: np.ndarray):
      return (pred > 0.5).cumprod(axis=1).sum(axis=1) - 1

    y_pred = prediction2label(net(x))
    return mean_squared_error(y.detach().numpy(), y_pred.detach().numpy())
