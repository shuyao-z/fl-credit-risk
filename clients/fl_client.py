# This file is modified from Jordan's code
from flwr.common.logger import log
from logging import DEBUG, INFO
import torch
import flwr as fl
import numpy as np
from collections import OrderedDict
from typing import List
import abc


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


class FLClient(fl.client.NumPyClient):
    def __init__(self, net, optimizer, x_train, y_train, x_test, y_test, cid=-1, num_epoch=256, batch_size=16):
        # super.__init__()
        self.net = net
        self.optimizer = optimizer
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.cid = cid
        self.num_epoch = num_epoch
        self.batch_size = batch_size

    @abc.abstractmethod
    def train(self, net, optimiser, x, y, num_epoch=256, batch_size=16):
        ...

    @abc.abstractmethod
    def test(self, net, x, y):
        ...

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        log(INFO, f"Client {self.cid} starts fit()")
        self.train(self.net, self.optimizer, self.x_train, self.y_train, self.num_epoch, self.batch_size)
        print(f"Client {self.cid} sends FitRes to server")
        return get_parameters(self.net), len(self.x_train), {}

    def evaluate(self, parameters, config):
        print(f'Client {self.cid} evaluate, config {config}')
        set_parameters(self.net, parameters)
        loss = self.test(self.net, self.x_test, self.y_test)
        mse = loss
        return float(loss), len(self.x_test), {"mse": float(mse)}

