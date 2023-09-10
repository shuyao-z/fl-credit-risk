# This file is modified from Jordan
import torch
import flwr as fl
from collections import OrderedDict
import json
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold, KFold

from clients.cr_client import CRClient
from clients.hcr_client import HCRClient
from clients.ts_client import TSClient
from net_archs import MLP, LSTMModel, MLP2, LSTMModel2

# dataset 1 MLP
#lr 5e-4 wd 1e-4
def mlp_simulation(split, x_train, y_train, x_test, y_test, dir='default', num_epoch=256, batch_size=16, num_rounds=15, lr=0.001, wd=0.00005, history_dir='./history/example.txt'):
  client_fn = fl_split(split, x_train, y_train, x_test, y_test, MLP(22, 20, layer_size=64, num_of_layers=5), CRClient, num_epoch, batch_size, lr, wd)
  return _run_simulation(client_fn, len(split), MLP(22, 20, layer_size=64, num_of_layers=5), upper_dir='model_checkpoints', dir=dir, num_rounds=num_rounds, history_dir=history_dir)

# dataset 1 LSTM
def lstm_simulation(split, x_train, y_train, x_test, y_test, dir='default', num_epoch=256, batch_size=4, num_rounds=20):
  client_fn = fl_split_lstm(split, x_train, y_train, x_test, y_test, LSTMModel(23, 21), CRClient, num_epoch, batch_size, 0.0001, 0.00001)
  return _run_simulation_lstm(client_fn, len(split), LSTMModel(23, 21), upper_dir='lstm_model_checkpoints', dir=dir, num_rounds=num_rounds)

# dataset 2 MLP
def mlp_simulation2(split, x_train, y_train, x_test, y_test, dir='default', num_epoch=16, batch_size=16, num_rounds=15, lr=0.000005, wd=0.00005,dropout=False,history_dir='./history/example.txt'):
  client_fn = fl_split(split, x_train, y_train, x_test, y_test, MLP(120, 1, layer_size=64, num_of_layers=2, dropout=dropout), HCRClient, num_epoch, batch_size, lr=lr, weight_decay=wd)
  return _run_simulation(client_fn, len(split), MLP(120, 1, layer_size=64, num_of_layers=2, dropout=dropout), upper_dir='model_checkpoints2', dir=dir, num_rounds=num_rounds, history_dir=history_dir)

# dataset 2 LSTM
def lstm_simulation2(split, nts_train, ts1_train, ts2_train, ts3_train, ts4_train, y_train, nts_test, ts1_test, ts2_test, ts3_test, ts4_test, y_test, dir='default', num_epoch=256, batch_size=16, num_rounds=15, dropout_rate=False):
  client_fn = fl_split2(split, nts_train, ts1_train, ts2_train, ts3_train, ts4_train, y_train, nts_test, ts1_test, ts2_test, ts3_test, ts4_test, y_test, LSTMModel2(layer_size=128, num_of_layers=2, dropout_rate=dropout_rate), TSClient, num_epoch, batch_size, 0.00001, 0.00001)
  return _run_simulation_lstm(client_fn, len(split), LSTMModel2(layer_size=128, num_of_layers=2, dropout_rate=dropout_rate), upper_dir='lstm_model_checkpoints2', dir=dir, num_rounds=num_rounds)

# dataset 3 MLP
def mlp_simulation3(split, x_train, y_train, x_test, y_test, dir='default', num_epoch=256, batch_size=32, num_rounds=1,layer_size=64, num_of_layers=2, dropout=False,history_dir='./history/example.txt'):
  client_fn = fl_split(split, x_train, y_train, x_test, y_test, MLP(10, 1, layer_size=layer_size, num_of_layers=num_of_layers, dropout=dropout), HCRClient, num_epoch, batch_size, lr=0.00001, weight_decay=0.00001)
  return _run_simulation(client_fn, len(split), MLP(10, 1, layer_size=layer_size, num_of_layers=num_of_layers, dropout=dropout), upper_dir='model_checkpoints3', dir=dir, num_rounds=num_rounds, history_dir=history_dir)


def fl_split2(split, nts_train, ts1_train, ts2_train, ts3_train, ts4_train, y_train, nts_test, ts1_test, ts2_test, ts3_test, ts4_test, y_test, net_arch, client, num_epoch, batch_size, lr, weight_decay):
  nts_eq = []
  ts1_eq = []
  ts2_eq = []
  ts3_eq = []
  ts4_eq = []
  y_eq = []
  # skf = StratifiedKFold(n_splits=10)
  n_splits = 50
  skf = KFold(n_splits=n_splits)
  skf.get_n_splits(nts_train, y_train)
  for i, (_, test_index) in enumerate(skf.split(nts_train, y_train)):
      nts_eq.append(nts_train[test_index])
      ts1_eq.append(ts1_train[test_index])
      ts2_eq.append(ts2_train[test_index])
      ts3_eq.append(ts3_train[test_index])
      ts4_eq.append(ts4_train[test_index])
      y_eq.append(y_train[test_index])

  x_split = []
  y_split = []

  acc = 0
  for s in split:
      x_split.append((
          torch.cat(nts_eq[acc:acc+int(s*n_splits)], 0),
          torch.cat(ts1_eq[acc:acc+int(s*n_splits)], 0),
          torch.cat(ts2_eq[acc:acc+int(s*n_splits)], 0),
          torch.cat(ts3_eq[acc:acc+int(s*n_splits)], 0),
          torch.cat(ts4_eq[acc:acc+int(s*n_splits)], 0)))
      y_split.append(torch.cat(y_eq[acc:acc+int(s*n_splits)], 0))
      acc += int(s*n_splits)

  def client_fn(cid):
      net = net_arch
      optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
      return TSClient(net,
                      optimizer,
                      x_train=x_split[int(cid)],
                      y_train=y_split[int(cid)],
                      x_test=(nts_test, ts1_test, ts2_test, ts3_test, ts4_test),
                      y_test=y_test,
                      cid=cid,
                      num_epoch=num_epoch,
                      batch_size=batch_size)
  return client_fn

def fl_split(split, x_train, y_train, x_test, y_test, net_arch, client, num_epoch, batch_size, lr, weight_decay):

    x_eq = []
    y_eq = []

    n_splits = 50
    #skf = StratifiedKFold(n_splits=n_splits)
    random_state = np.random.randint(1000)
    skf = KFold(n_splits=n_splits,shuffle=True, random_state=random_state)
    skf.get_n_splits(x_train, y_train)

    for i, (_, test_index) in enumerate(skf.split(x_train, y_train)):
        x_eq.append(x_train[test_index])
        y_eq.append(y_train[test_index])

    x_split = []
    y_split = []


    acc = 0
    for s in split:
        x_split.append(torch.cat(x_eq[acc:acc+int(s*n_splits)], 0))
        y_split.append(torch.cat(y_eq[acc:acc+int(s*n_splits)], 0))
        acc += int(s*n_splits)

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    x_test = x_test.to(DEVICE)
    y_test = y_test.to(DEVICE)

    for i in range(len(x_split)):
        ros = RandomOverSampler(random_state=42)
        #X_train, y_train = ros.fit_resample(x_split[i], y_split[i])

        X_train_testing, y_train_testing = ros.fit_resample(x_split[i], y_split[i])

        #X_train_testing, y_train_testing = ros.fit_resample(x_train, y_train)
        #X_train_testing = torch.from_numpy(X_train_testing)
        y_train_testing = torch.reshape(torch.tensor(y_train_testing), (-1, 1))

        x_split[i] = torch.from_numpy(X_train_testing).to(DEVICE)

        print(f'Client {i} trainset shape {x_split[i].shape}')
        y_split[i] = y_train_testing.to(DEVICE)

    def client_fn(cid):
        # net = net_arch(22, 21)
        net = net_arch.to(DEVICE)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

        return client(net,
                      optimizer,
                      x_train=x_split[int(cid)],
                      y_train=y_split[int(cid)],
                      #x_train=X_train_testing,
                      #y_train=y_train_testing,
                      x_test=x_test,
                      y_test=y_test,
                      #x_train=torch.load('./dataset3/xtrain.pt'),
                      #y_train=torch.load('./dataset3/ytrain.pt'),
                      #x_test=torch.load('./dataset3/xtest.pt'),
                      #y_test=torch.load('./dataset3/ytest.pt'),
                      cid=cid,
                      num_epoch=num_epoch,
                      batch_size=batch_size)
    return client_fn


def fl_split_lstm(split, x_train, y_train, x_test, y_test, net_arch, client, num_epoch, batch_size, lr, weight_decay):

    x_eq = []
    y_eq = []

    n_splits = 50
    skf = KFold(n_splits=n_splits)
    skf.get_n_splits(x_train, y_train)

    for i, (_, test_index) in enumerate(skf.split(x_train, y_train)):
        x_eq.append(x_train[test_index])
        y_eq.append(y_train[test_index])

    x_split = []
    y_split = []


    acc = 0
    for s in split:
        x_split.append(torch.cat(x_eq[acc:acc+int(s*n_splits)], 0))
        y_split.append(torch.cat(y_eq[acc:acc+int(s*n_splits)], 0))
        acc += int(s*n_splits)


    for i in range(len(x_split)):
        ros = RandomOverSampler(random_state=42)

        X_resampled, y_resampled = ros.fit_resample(x_split[i].reshape(-1, 3 * 23), y_split[i])
        X_train = X_resampled.reshape(X_resampled.shape[0], 3, 23)

        #y_train_testing = torch.reshape(torch.tensor(y_train_testing), (-1, 1))
        print(f'Client {i} trainset shape {x_split[i].shape}')
        #x_split[i] = torch.from_numpy(X_train)
        #y_split[i] = torch.from_numpy(y_resampled)

    def client_fn(cid):
        net = net_arch
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

        return client(net,
                      optimizer,
                      x_train=x_split[int(cid)],
                      y_train=y_split[int(cid)],
                      x_test=x_test,
                      y_test=y_test,
                      cid=cid,
                      num_epoch=num_epoch,
                      batch_size=batch_size)
    return client_fn

def _run_simulation(client_fn, num_clients, net_arch, upper_dir='default', dir='default', num_rounds=15, print_hist = True, history_dir='./history/example.txt'):
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    #print('num of rounds', num_rounds)

    print(f"FL started training on {DEVICE}")

    num_gpus = torch.cuda.device_count()

    print(f"Number of available GPUs: {num_gpus}")

    model_fl = net_arch.to(DEVICE)

    class SaveModelStrategy(fl.server.strategy.FedAvg):
        def aggregate_fit(
            self,
            server_round,
            results,
            failures,
        ):

            print('agrregate_fit results list length is ', len(results))

            # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
            aggregated_parameters, aggregated_metrics = super(
            ).aggregate_fit(server_round, results, failures)

            ndarrays = fl.common.parameters_to_ndarrays(
                    aggregated_parameters)


            if aggregated_parameters is not None:
                print(f"Saving round {server_round} aggregated_parameters...")

                # Convert `Parameters` to `List[np.ndarray]`
                aggregated_ndarrays = fl.common.parameters_to_ndarrays(
                    aggregated_parameters)

                # Convert `List[np.ndarray]` to PyTorch`state_dict`
                params_dict = zip(
                    model_fl.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict(
                    {k: torch.tensor(v) for k, v in params_dict})
                model_fl.load_state_dict(state_dict, strict=True)

                # Save the model
                torch.save(model_fl.state_dict(
                ), f"{upper_dir}/{dir}/model_round_{server_round}.pth")

            return aggregated_parameters, aggregated_metrics

        def aggregate_evaluate(
                self,
                server_round,
                results,
                failures,
        ):
            print('agrregate_evaluate results list length is ', len(results))

            """Aggregate evaluation losses using weighted average."""
            if not results:
                print('results is None')
                return None, {}

            loss, mse = super().aggregate_evaluate(server_round, results, failures)

            print('server round is ', server_round)
            print('loss is ', loss)

            return loss, mse

        def configure_evaluate(
                self,
                server_round,
                parameters: Parameters,
                client_manager,
    ) : #-> List[Tuple[ClientProxy, EvaluateIns]]
            instruction = super().configure_evaluate(server_round, parameters, client_manager)
            #print('configure_evaluate length', len(instruction))
            return instruction

    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # Multiply accuracy of each client by number of examples used
        total_sum = 0
        for value, metric in metrics:
            total_sum += metric['mse']

        # Aggregate and return custom metric (weighted average)
        return {"mse": total_sum}

    # Create FedAvg strategy
    strategy = SaveModelStrategy(
        fraction_fit=0.5,  # was 0.5
        fraction_evaluate=0.5, # was 0.5
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=weighted_average
    )

    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    client_resources = None

    if DEVICE.type == "cuda":
        client_resources = {"num_gpus": num_gpus}

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        clients_ids=[str(x) for x in range(num_clients)],
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args={'log_to_driver': False}
    )

    if print_hist:
        print(history)
        print(history.losses_distributed)

    # with open(history_dir, 'w') as file:
    #     for entry in history.losses_distributed:
    #         line = f"{entry[0]}, {entry[1]}\n"  # Convert tuple to a comma-separated string
    #         file.write(line)
    #
    with open(history_dir, 'w') as file:
        # Step 3: Write the history data to the CSV file
        for entry in history.losses_distributed:
            line = f"{entry[0]},{entry[1]}\n"  # Convert tuple to a comma-separated string
            file.write(line)

    latest_round_file = f'{upper_dir}/{dir}/model_round_{num_rounds}.pth'
    state_dict = torch.load(latest_round_file)
    model_fl.load_state_dict(state_dict)

    return model_fl


def _run_simulation_lstm(client_fn, num_clients, net_arch, upper_dir='default', dir='default', num_rounds=15, print_hist = True, history_dir='./history/example.txt'):
    model_fl = net_arch
    class SaveModelStrategy(fl.server.strategy.FedAvg):
        def aggregate_fit(
            self,
            server_round,
            results,
            failures,
        ):

            print('agrregate_fit results list length is ', len(results))

            # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
            aggregated_parameters, aggregated_metrics = super(
            ).aggregate_fit(server_round, results, failures)

            ndarrays = fl.common.parameters_to_ndarrays(
                    aggregated_parameters)


            if aggregated_parameters is not None:
                print(f"Saving round {server_round} aggregated_parameters...")

                # Convert `Parameters` to `List[np.ndarray]`
                aggregated_ndarrays = fl.common.parameters_to_ndarrays(
                    aggregated_parameters)

                # Convert `List[np.ndarray]` to PyTorch`state_dict`
                params_dict = zip(
                    model_fl.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict(
                    {k: torch.tensor(v) for k, v in params_dict})
                model_fl.load_state_dict(state_dict, strict=True)

                # Save the model
                torch.save(model_fl.state_dict(
                ), f"{upper_dir}/{dir}/model_round_{server_round}.pth")

            return aggregated_parameters, aggregated_metrics

        def aggregate_evaluate(
                self,
                server_round,
                results,
                failures,
        ):
            print('agrregate_evaluate results list length is ', len(results))

            """Aggregate evaluation losses using weighted average."""
            if not results:
                print('results is None')
                return None, {}

            loss, mse = super().aggregate_evaluate(server_round, results, failures)

            print('server round is ', server_round)
            print('loss is ', loss)

            return loss, mse

        def configure_evaluate(
                self,
                server_round,
                parameters: Parameters,
                client_manager,
    ) : #-> List[Tuple[ClientProxy, EvaluateIns]]
            instruction = super().configure_evaluate(server_round, parameters, client_manager)
            #print('configure_evaluate length', len(instruction))
            return instruction

    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # Multiply accuracy of each client by number of examples used
        total_sum = 0
        for value, metric in metrics:
            total_sum += metric['mse']

        # Aggregate and return custom metric (weighted average)
        return {"mse": total_sum}

    # Create FedAvg strategy
    strategy = SaveModelStrategy(
        fraction_fit=0.5,  # was 0.5
        fraction_evaluate=0.5, # was 0.5
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=weighted_average
    )

    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    client_resources = None

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        clients_ids=[str(x) for x in range(num_clients)],
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args={'log_to_driver': False}
    )

    if print_hist:
        print(history)
        print(history.losses_distributed)

    with open(history_dir, 'w') as file:
        # Step 3: Write the history data to the CSV file
        for entry in history.losses_distributed:
            line = f"{entry[0]},{entry[1]}\n"  # Convert tuple to a comma-separated string
            file.write(line)

    latest_round_file = f'{upper_dir}/{dir}/model_round_{num_rounds}.pth'
    state_dict = torch.load(latest_round_file)
    model_fl.load_state_dict(state_dict)

    return model_fl


import flwr as fl
from typing import Dict
from torch.utils.data import DataLoader
from fl_server import FL_Server
from net_archs import CNN
from flwr.server.history import History
from clients.xgb_client import xgb_client
from clients.xgb_client import test as binary_test
from flwr.server.strategy import FedXgbNnAvg
from clients.xgb_multi_client import xgb_multi_client, test

from flwr.server.client_manager import SimpleClientManager
import functools
from flwr.server.app import ServerConfig
from flwr.common import parameters_to_ndarrays
from trees import do_fl_partitioning, tree_encoding_loader
import torch, torch.nn as nn
from flwr.common import Scalar
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional, Tuple, Union
from flwr.common.typing import Parameters, Metrics
from xgboost import XGBClassifier, XGBRegressor

def print_model_layers(model: nn.Module) -> None:
    print(model)
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())


def serverside_eval(
    server_round: int,
    parameters: Tuple[
        Parameters,
        Union[
            Tuple[XGBClassifier, int],
            Tuple[XGBRegressor, int],
            List[Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]],
        ],
    ],
    config: Dict[str, Scalar],
    testloader: DataLoader,
    batch_size: int,
    client_tree_num: int,
    client_num: int,
    task_type: str = 'binary'
) -> Tuple[float, Dict[str, float]]:
    """An evaluation function for centralized/serverside evaluation over the entire test set."""
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('serverside task_type', task_type)
    device = "cpu"
    model = CNN(client_tree_num=client_tree_num, client_num=client_num, task_type=task_type)
    # print_model_layers(model)

    model.set_weights(parameters_to_ndarrays(parameters[0]))
    model.to(device)

    trees_aggregated = parameters[1]

    testloader = tree_encoding_loader(
        testloader, batch_size, trees_aggregated, client_tree_num, client_num, task_type=task_type
    )

    if task_type == 'multiclass':
        mse, result, _ = test(
             model, client_tree_num, client_num,testloader, device=device, log_progress=False, task_type=task_type
        )
    elif task_type == 'binary':
        mse, result, _ = binary_test(
             model, testloader, device=device, log_progress=False, task_type=task_type
        )

    # torch.save(trees_aggregated, './trees.pt')
    # torch.save(model, './model.pt')

    print(
        f"Evaluation on the server: test_mse={mse:.4f}, test_auc={result:.4f}"
    )
    return mse, {"auc": result}

def start_experiment(
    split: list,
    trainset: Dataset,
    testset: Dataset,
    max_depth: int = 5,
    min_child_weight: int = 1,
    subsample: float=1.0,
    num_rounds: int = 5,
    client_tree_num: int = 50,
    client_pool_size: int = 5,
    num_iterations: int = 100,
    fraction_fit: float = 1.0,
    min_fit_clients: int = 2,
    batch_size: int = 32,
    val_ratio: float = 0.1,
    client_num: int = 10,
    task_type: str = 'binary',
    lr=0.01,
    colsample_bytree: float=1.0
) -> History:
    client_resources = {"num_cpus": 1}  # 2 clients per CPU

    # Partition the dataset into subsets reserved for each client.
    # - 'val_ratio' controls the proportion of the (local) client reserved as a local test set
    # (good for testing how the final model performs on the client's local unseen data)
    trainloaders, valloaders, testloader = do_fl_partitioning(
        split,
        trainset,
        testset,
        batch_size="whole",
        pool_size=client_pool_size,
        val_ratio=val_ratio,
    )
    print(
        f"Data partitioned across {client_pool_size} clients"
        f" and {val_ratio} of local dataset reserved for validation."
    )

    # Configure the strategy
    def fit_config(server_round: int) -> Dict[str, Scalar]:
        print(f"Configuring round {server_round}")
        return {
            "num_iterations": num_iterations,
            "batch_size": batch_size,
        }

    # FedXgbNnAvg
    strategy = FedXgbNnAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_fit if val_ratio > 0.0 else 0.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_fit_clients,
        min_available_clients=client_pool_size,  # all clients should be available
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=(lambda r: {"batch_size": batch_size}),
        evaluate_fn=functools.partial(
            serverside_eval,
            testloader=testloader,
            batch_size=batch_size,
            client_tree_num=client_tree_num,
            client_num=client_num,
            task_type=task_type
        ),
        accept_failures=False,
    )

    print(
        f"FL experiment configured for {num_rounds} rounds with {client_pool_size} client in the pool."
    )
    print(
        f"FL round will proceed with {fraction_fit * 100}% of clients sampled, at least {min_fit_clients}."
    )

    def client_fn(cid: str) -> fl.client.Client:
        """Creates a federated learning client"""
        if task_type=='binary':
            print("binary client")
            if val_ratio > 0.0 and val_ratio <= 1.0:
                return xgb_client(
                    trainloaders[int(cid)],
                    valloaders[int(cid)],
                    client_tree_num,
                    client_pool_size,
                    cid,
                    max_depth=max_depth,
                    min_child_weight=min_child_weight,
                    subsample=subsample,
                    log_progress=False,
                    task_type=task_type,
                    lr=lr,
                    colsample_bytree=colsample_bytree,
                )
            else:
                return xgb_client(
                    trainloaders[int(cid)],
                    None,
                    client_tree_num,
                    client_pool_size,
                    cid,
                    max_depth=max_depth,
                    min_child_weight=min_child_weight,
                    subsample=subsample,
                    log_progress=False,
                    task_type=task_type,
                    lr=lr,
                    colsample_bytree=colsample_bytree,
                )

        elif task_type == 'multiclass':
            print("multiclass client")
            if val_ratio > 0.0 and val_ratio <= 1.0:
                return xgb_multi_client(
                    trainloaders[int(cid)],
                    valloaders[int(cid)],
                    client_tree_num,
                    client_pool_size,
                    cid,
                    max_depth=max_depth,
                    min_child_weight=min_child_weight,
                    subsample=subsample,
                    log_progress=False,
                    task_type=task_type,
                    lr=lr,
                    colsample_bytree=colsample_bytree,
                )
            else:
                return xgb_multi_client(
                    trainloaders[int(cid)],
                    None,
                    client_tree_num,
                    client_pool_size,
                    cid,
                    max_depth=max_depth,
                    min_child_weight=min_child_weight,
                    subsample=subsample,
                    log_progress=False,
                    task_type=task_type,
                    lr=lr,
                    colsample_bytree=colsample_bytree,
                )

    # Start the simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        server=FL_Server(client_manager=SimpleClientManager(), strategy=strategy),
        num_clients=client_pool_size,
        client_resources=client_resources,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    print(history)
    print(history.losses_centralized)

    return history

