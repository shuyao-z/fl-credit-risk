# This file is modified from https://github.com/adap/flower/tree/main/examples/quickstart-xgboost-horizontal

import torch, torch.nn as nn
from trees import construct_tree_from_loader, tree_encoding_loader
from torch.utils.data import DataLoader
from typing import List, Tuple, Union
from xgboost import XGBClassifier, XGBRegressor
from tqdm import trange, tqdm
import flwr as fl
from sklearn.metrics import mean_squared_error,  roc_auc_score
import numpy as np

from net_archs import CNN

from flwr.common.typing import Parameters
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetPropertiesIns,
    GetPropertiesRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    Code,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)


class xgb_multi_client(fl.client.Client):
    def __init__(
        self,
        trainloader: DataLoader,
        valloader: DataLoader,
        client_tree_num: int,
        client_num: int,
        cid: str,
        max_depth,
        min_child_weight,
        subsample,
        log_progress: bool = True,
        n_channel: int = 64,
        task_type: str = 'binary',
        lr=0.01,
        colsample_bytree: float = 1.0
    ):

        self.task_type = task_type
        self.cid = cid
        self.tree = construct_tree_from_loader(trainloader, client_tree_num, self.task_type,
                                               max_depth=max_depth, min_child_weight=min_child_weight,
                                               subsample=subsample, colsample_bytree=colsample_bytree)
        self.trainloader_original = trainloader
        self.valloader_original = valloader
        self.trainloader = None
        self.valloader = None
        self.client_tree_num = client_tree_num
        self.client_num = client_num
        self.properties = {"tensor_type": "numpy.ndarray"}
        self.log_progress = log_progress
        self.lr = lr

        # instantiate model
        self.net = CNN(n_channel=n_channel, client_tree_num=client_tree_num, client_num=client_num, task_type=task_type)

        # determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        return GetPropertiesRes(properties=self.properties)

    def get_parameters(
        self, ins: GetParametersIns
    ) -> Tuple[
        GetParametersRes, Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]
    ]:
        return [
            GetParametersRes(
                status=Status(Code.OK, ""),
                parameters=ndarrays_to_parameters(self.net.get_weights()),
            ),
            (self.tree, int(self.cid)),
        ]

    def set_parameters(
        self,
        parameters: Tuple[
            Parameters,
            Union[
                Tuple[XGBClassifier, int],
                Tuple[XGBRegressor, int],
                List[Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]],
            ],
        ],
    ) -> Union[
        Tuple[XGBClassifier, int],
        Tuple[XGBRegressor, int],
        List[Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]],
    ]:
        self.net.set_weights(parameters_to_ndarrays(parameters[0]))
        return parameters[1]

    def fit(self, fit_params: FitIns) -> FitRes:
        # Process incoming request to train
        num_iterations = fit_params.config["num_iterations"]
        batch_size = fit_params.config["batch_size"]
        aggregated_trees = self.set_parameters(fit_params.parameters)
        length = len(aggregated_trees)
        print(f'Client {self.cid} aggregated_trees length {length}')

        if type(aggregated_trees) is list:
            print("Client " + self.cid + ": recieved", len(aggregated_trees), "trees")
        else:
            print("Client " + self.cid + ": only had its own tree")

        self.trainloader = tree_encoding_loader(
            self.trainloader_original,
            batch_size,
            aggregated_trees,
            self.client_tree_num,
            self.client_num,
            task_type=self.task_type
        )
        self.valloader = tree_encoding_loader(
            self.valloader_original,
            batch_size,
            aggregated_trees,
            self.client_tree_num,
            self.client_num,
            task_type=self.task_type
        )

        # num_iterations = None special behaviour: train(...) runs for a single epoch, however many updates it may be
        num_iterations = num_iterations or len(self.trainloader)

        # Train the model
        print(f"Client {self.cid}: training for {num_iterations} iterations/updates")
        self.net.to(self.device)
        train_loss, train_auc, num_examples = train(
            self.net,
            self.client_tree_num,
            self.client_num,
            self.trainloader,
            device=self.device,
            num_iterations=num_iterations,
            log_progress=self.log_progress,
            task_type=self.task_type,
            lr= self.lr
        )

        # Return training information: model, number of examples processed and metrics
        return FitRes(
            status=Status(Code.OK, ""),
            parameters=self.get_parameters(fit_params.config),
            num_examples=num_examples,
            metrics={"loss": train_loss, "auc": train_auc},
        )


    def evaluate(self, eval_params: EvaluateIns) -> EvaluateRes:
        # Process incoming request to evaluate
        self.set_parameters(eval_params.parameters)

        # Evaluate the model
        self.net.to(self.device)
        loss, result, num_examples = test(
            self.net,
            self.client_tree_num,
            self.client_num,
            self.valloader,
            device=self.device,
            log_progress=self.log_progress,
            task_type=self.task_type
        )

        # Return evaluation information
        print(
            f"Client {self.cid}: evaluation on {num_examples} examples: loss={loss:.4f}, auc={result:.4f}"
        )
        return EvaluateRes(
            status=Status(Code.OK, ""),
            loss=loss,
            num_examples=num_examples,
            metrics={"auc": result},
        )

def train(
    net: CNN,
    client_tree_num,
    client_num,
    trainloader: DataLoader,
    device: torch.device,
    num_iterations: int,
    log_progress: bool = True,
    task_type: str = 'binary',
    lr = 0.001
) -> Tuple[float, float, int]:

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.95, 0.999))

    def cycle(iterable):
        """Repeats the contents of the train loader, in case it gets exhausted in 'num_iterations'."""
        while True:
            for x in iterable:
                yield x

    # Train the network
    net.train()
    total_loss, total_auc, n_samples = 1, 1, 1
    pbar = (
        tqdm(iter(cycle(trainloader)), total=num_iterations, desc=f"TRAIN")
        if log_progress
        else iter(cycle(trainloader))
    )

    # Unusually, this training is formulated in terms of number of updates/iterations/batches processed
    # by the network. This will be helpful later on, when partitioning the data across clients: resulting
    # in differences between dataset sizes and hence inconsistent numbers of updates per 'epoch'.
    for i, data in zip(range(num_iterations), pbar):

        tree_outputs, labels = data[0].to(device), data[1].to(device)
        torch.save(tree_outputs, './dataset/xgb_X_train.pt')
        torch.save(labels, './dataset/xgb_y_train.pt')

        X_train = torch.empty((0, 1, 20*client_num*client_tree_num))
        for row in tree_outputs:
            new_row = torch.transpose(row.view(-1, 20), 0, 1)
            new = new_row.reshape(-1, 20*client_num*client_tree_num).unsqueeze(0)
            X_train = torch.cat((X_train, new), dim=0)

        for tree_output, label in zip(X_train, labels):
            optimizer.zero_grad()
            outputs = net(tree_output)
            loss = ordinal_criterion(outputs, label)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

    auc = 0

    if log_progress:
        pbar.set_postfix(
            {
                "train_loss": total_loss / n_samples,
                "train_auc": auc,
            }
        )

    if log_progress:
        print("\n")

    return total_loss / n_samples, auc, n_samples


def test(
    net: CNN,
    client_tree_num,
    client_num,
    testloader: DataLoader,
    device: torch.device,
    log_progress: bool = True,
    task_type: str = 'binary'
) -> Tuple[float, float, int]:
    """Evaluates the network on test data."""
    print('test ', task_type)

    total_loss, total_result, n_samples = 0.0, 0.0, 0
    net.eval()

    if log_progress:
        print("\n")

    mse, auc = 0.0, 0.0

    with torch.no_grad():
        pbar = tqdm(testloader, desc="TEST") if log_progress else testloader
        for data in pbar:
            tree_outputs, labels = data[0].to(device), data[1].to(device)
            torch.save(tree_outputs, './dataset/xgb_X_test.pt')
            torch.save(labels, './dataset/xgb_y_test.pt')

            X_test = torch.empty((0, 1, 20*client_num*client_tree_num))
            for row in tree_outputs:
                new_row = torch.transpose(row.view(-1, 20), 0, 1)
                new = new_row.reshape(-1, 20*client_num*client_tree_num).unsqueeze(0)
                X_test = torch.cat((X_test, new), dim=0)


    y_pred = net(X_test)

    if y_pred.shape[1] == 20:
        y_pred = prediction2label(net(X_test))
    else:
        y_pred = torch.clip(y_pred.round(), min=0, max=20)

    y_default_test = np.where(labels.detach().numpy() > 9, 1, 0)
    y_default_pred = np.where(y_pred.detach().numpy() > 9, 1, 0)

    mse = mean_squared_error(labels.detach().numpy(), y_pred.detach().numpy())
    auc = roc_auc_score(y_default_test, y_default_pred)

    return mse, auc, n_samples


def prediction2label(pred: np.ndarray):
  """Convert ordinal predictions to class labels, e.g.

  [0.9, 0.1, 0.1, 0.1] -> 0
  [0.9, 0.9, 0.1, 0.1] -> 1
  [0.9, 0.9, 0.9, 0.1] -> 2
  etc.
  """
  return (pred > 0.5).cumprod(axis=1).sum(axis=1) - 1


def ordinal_criterion(predictions, targets):
    modified_target = torch.zeros_like(predictions)
    for i, target in enumerate(targets):
        modified_target[i, 0:int(target) + 1] = 1

    return nn.MSELoss(reduction='none')(predictions, modified_target).sum(axis=1)

