# This file is modified from https://github.com/adap/flower/tree/main/examples/quickstart-xgboost-horizontal

import torch, torch.nn as nn
from trees import construct_tree_from_loader, tree_encoding_loader
from torch.utils.data import DataLoader
from typing import List, Tuple, Union
from xgboost import XGBClassifier, XGBRegressor
from tqdm import tqdm
import flwr as fl
from sklearn.metrics import mean_squared_error,  roc_auc_score
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


class xgb_client(fl.client.Client):
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
            self.trainloader,
            device=self.device,
            num_iterations=num_iterations,
            log_progress=self.log_progress,
            task_type=self.task_type,
            lr=self.lr
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
    trainloader: DataLoader,
    device: torch.device,
    num_iterations: int,
    log_progress: bool = True,
    lr=0.01,
    task_type = 'binary'
) -> Tuple[float, float, int]:
    # Define loss and optimizer

    criterion = nn.MSELoss()  # need to confirm others use mse
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))

    def cycle(iterable):
        """Repeats the contents of the train loader, in case it gets exhausted in 'num_iterations'."""
        while True:
            for x in iterable:
                yield x

    # Train the network
    net.train()
    pbar = (
        tqdm(iter(cycle(trainloader)), total=num_iterations, desc=f"TRAIN")
        if log_progress
        else iter(cycle(trainloader))
    )

    # Unusually, this training is formulated in terms of number of updates/iterations/batches processed
    # by the network. This will be helpful later on, when partitioning the data across clients: resulting
    # in differences between dataset sizes and hence inconsistent numbers of updates per 'epoch'.
    y_test = torch.empty((0, 1))
    y_pred = torch.empty((0, 1))

    total_loss, total_result, n_samples = 0.0, 0.0, 0

    for i, data in zip(range(num_iterations), pbar):
        tree_outputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = net(tree_outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_samples += labels.size(0)

        y_test = torch.cat((torch.tensor(y_test), torch.tensor(labels)), dim=0)
        y_pred = torch.cat((torch.tensor(y_pred), torch.tensor(outputs)), dim=0)


        y_pred = y_pred.detach().numpy()
        y_test = y_test.detach().numpy()


    if log_progress:
        print("\n")

    mse = 1
    auc = 1

    return mse / n_samples, auc, n_samples


def test(
    net: CNN,
    testloader: DataLoader,
    device: torch.device,
    log_progress: bool = True,
    task_type: str = 'binary'
) -> Tuple[float, float, int]:
    """Evaluates the network on test data."""

    criterion = nn.MSELoss()
    total_loss, total_result, n_samples = 0.0, 0.0, 0
    net.eval()

    y_test = torch.empty((0, 1))
    y_pred = torch.empty((0, 1))

    with torch.no_grad():
        pbar = tqdm(testloader, desc="TEST") if log_progress else testloader
        for data in pbar:
            tree_outputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(tree_outputs)
            n_samples += labels.size(0)

            total_loss += criterion(outputs, labels).item()
            n_samples += labels.size(0)

            y_test = torch.cat((y_test, labels), dim=0)
            y_pred = torch.cat((y_pred, outputs), dim=0)

    if log_progress:
        print("\n")

    y_pred = y_pred.detach().numpy()
    y_test = y_test.detach().numpy()

    mse = mean_squared_error(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    print(f'testset mse {mse} and auc {auc}')

    return mse, auc, n_samples
