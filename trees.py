from imblearn.over_sampling import RandomOverSampler
from matplotlib import pyplot as plt  # pylint: disable=E0401
from typing import Union
import xgboost as xgb
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier, XGBRegressor
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset, random_split
from flwr.common import NDArray, NDArrays

def plot_xgbtree(tree: Union[XGBClassifier, XGBRegressor], n_tree: int) -> None:
    """Visualize the built xgboost tree."""
    xgb.plot_tree(tree, num_trees=n_tree)
    plt.rcParams["figure.figsize"] = [50, 10]
    plt.show()

def construct_tree(
    dataset: Dataset,
    label: NDArray,
    n_estimators: int,
    learning_rate: float = 0.3, # 0.1 for binary
    gamma = 0,
    max_depth: int = 5,
    min_child_weight = 3,
    subsample = 0.8,
    colsample_bytree = 1.0,
    task_type: str = 'binary'
) -> Union[XGBClassifier, XGBRegressor]:

    params = {}

    if task_type == 'binary':
        #print('construct binary xgb classifier')
        params = {
            'n_jobs': -1,
            'objective': 'binary:logistic',
            'eta': learning_rate,
            'max_depth': max_depth,
            'gamma': gamma,
            'n_estimators': n_estimators,
            'min_child_weight': min_child_weight,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'eval_metric': 'auc',
        }

    if task_type == 'multiclass':
        #print('construct multiclass xgb classifier')
        params = {
            'n_jobs': -1,
            'objective': 'multi:softmax',
            'eta': learning_rate,
            'max_depth': max_depth,
            'gamma': gamma,
            'n_estimators': n_estimators,
            'min_child_weight': min_child_weight,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'eval_metric': 'auc',
            'num_class': 20
        }

    model = xgb.XGBClassifier(**params)
    model.fit(dataset, label)

    #print('tree output shape', model.predict(dataset).shape)

    return model


def construct_tree_from_loader(
    dataset_loader: DataLoader, n_estimators: int, task_type: str,
    max_depth: int=5,
    min_child_weight: int=3,
    subsample: float=1.0,
    colsample_bytree: float=1.0
) -> Union[XGBClassifier, XGBRegressor]:
    """Construct a xgboost tree form tabular dataset loader."""
    for dataset in dataset_loader:
        data, label = dataset[0], dataset[1]

    return construct_tree(data, label, n_estimators,
                          max_depth=max_depth,
                          min_child_weight=min_child_weight,
                          subsample=subsample,
                          colsample_bytree=colsample_bytree,
                          task_type=task_type)


def single_tree_prediction(
    tree: Union[XGBClassifier, XGBRegressor], n_tree: int, dataset: NDArray, task_type: str='binary'
) -> Optional[NDArray]:
    """Extract the prediction result of a single tree in the xgboost tree
    ensemble."""
    # How to access a single tree
    # https://github.com/bmreiniger/datascience.stackexchange/blob/master/57905.ipynb
    num_t = len(tree.get_booster().get_dump())
    if n_tree > num_t:
        print(
            "The tree index to be extracted is larger than the total number of trees."
        )
        return None

    if task_type == 'binary':
        output = tree.predict(  # type: ignore
            dataset, iteration_range=(n_tree, n_tree + 1), output_margin=True
        )
    else:
        output = tree.predict(dataset, iteration_range=(n_tree, n_tree + 1), output_margin=True)

    #print('output in single tree prediction', output)

    return output


class TreeDataset(Dataset):
    def __init__(self, data: NDArray, labels: NDArray) -> None:
        self.labels = labels
        self.data = data
        #print('initial label shape', self.labels.shape)
        #print('initial data shape', self.data.shape)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[int, NDArray]:
        label = self.labels[idx]
        #print(self.data[0])
        # if idx == 1:
        #     print('data shape',self.data.shape)
        #     print('label shape', self.labels.shape)
        data = self.data[idx, :]
        sample = {0: data, 1: label}
        return sample

def get_dataloader(
    dataset: Dataset, partition: str, batch_size: Union[int, str]
) -> DataLoader:
    if batch_size == "whole":
        batch_size = len(dataset)
    return DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, shuffle=(partition == "train")
    )

def do_fl_partitioning(
    split,
    trainset: Dataset,
    testset: Dataset,
    pool_size: int,
    batch_size: Union[int, str],
    val_ratio: float = 0.0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    x_train = torch.tensor(trainset.data)
    y_train = torch.tensor(trainset.labels)

    x_eq = []
    y_eq = []

    n_splits = 50
    skf = StratifiedKFold(n_splits=n_splits)
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

    #print(x_split[0].size())

    for i in range(len(x_split)):
        ros = RandomOverSampler(random_state=42)
        X_train_testing, y_train_testing = ros.fit_resample(x_split[i], y_split[i])

        x_split[i] = torch.from_numpy(X_train_testing)
        #y_split[i] = torch.from_numpy(y_train_testing)
        #
        # y_train_testing = torch.reshape(torch.tensor(y_train_testing), (-1, 1))
        #

        y_split[i] = y_train_testing

    #print(x_split[0].size())

    datasets = []
    for index, item in enumerate(x_split):
        ds = TreeDataset(x_split[index], y_split[index])
        datasets.append(ds)

    trainloaders = []

    for ds_train in datasets:
        print(ds_train.data.shape)
        trainloaders.append(get_dataloader(ds_train,"train", batch_size))

    valloaders = None
    testloader = get_dataloader(testset, "test", batch_size)

    return trainloaders, valloaders, testloader
#
# def do_fl_partitioning(
#     trainset: Dataset,
#     testset: Dataset,
#     pool_size: int,
#     batch_size: Union[int, str],
#     val_ratio: float = 0.0,
# ) -> Tuple[DataLoader, DataLoader, DataLoader]:
#     # Split training set into `num_clients` partitions to simulate different local datasets
#     partition_size = len(trainset) // pool_size
#     lengths = [partition_size] * pool_size
#     if sum(lengths) != len(trainset):
#         lengths[-1] = len(trainset) - sum(lengths[0:-1])
#     datasets = random_split(trainset, lengths, torch.Generator().manual_seed(0))
#
#     # Split each partition into train/val and create DataLoader
#     trainloaders = []
#     valloaders = []
#     for ds in datasets:
#         len_val = int(len(ds) * val_ratio)
#         len_train = len(ds) - len_val
#         lengths = [len_train, len_val]
#         ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(0))
#         trainloaders.append(get_dataloader(ds_train, "train", batch_size))
#         if len_val != 0:
#             valloaders.append(get_dataloader(ds_val, "val", batch_size))
#         else:
#             valloaders = None
#     testloader = get_dataloader(testset, "test", batch_size)
#     torch.save(testloader, './testloader.pt')
#     return trainloaders, valloaders, testloader


def tree_encoding(  # pylint: disable=R0914
    trainloader: DataLoader,
    client_trees: Union[
        Tuple[XGBClassifier, int],
        Tuple[XGBRegressor, int],
        List[Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]],
    ],
    client_tree_num: int,
    client_num: int,
    task_type: str = 'binary'
) -> Optional[Tuple[NDArray, NDArray]]:
    """Transform the tabular dataset into prediction results using the
    aggregated xgboost tree ensembles from all clients."""
    if trainloader is None:
        return None

    for local_dataset in trainloader:
        x_train, y_train = local_dataset[0], local_dataset[1]

    # print('x_train.shape', x_train.shape)
    # print('y_train.shape', y_train.shape)


    x_train_enc = np.zeros((x_train.shape[0], client_num * client_tree_num))
    if task_type == 'multiclass':
        x_train_enc = np.zeros((x_train.shape[0], client_num * client_tree_num * 20))

    x_train_enc = np.array(x_train_enc, copy=True)

    #print('x_train_enc', x_train_enc.shape)

    temp_trees: Any = None
    if isinstance(client_trees, list) is False:
        temp_trees = [client_trees[0]] * client_num
    elif isinstance(client_trees, list) and len(client_trees) != client_num:
        temp_trees = [client_trees[0][0]] * client_num
    else:
        cids = []
        temp_trees = []
        for i, _ in enumerate(client_trees):
            temp_trees.append(client_trees[i][0])  # type: ignore
            cids.append(client_trees[i][1])  # type: ignore
        sorted_index = np.argsort(np.asarray(cids))
        temp_trees = np.asarray(temp_trees)[sorted_index]

    for i, _ in enumerate(temp_trees):
        for j in range(client_tree_num):
            test = single_tree_prediction(
                temp_trees[i], j, x_train, task_type=task_type
            )
            #print('temp trees shape', temp_trees[i].shape)
            #print('x_train shape', x_train.shape)
            #print('single tree prediction shape',test.shape)
            #print('single_tree_prediction result', test[0])
            #print('x_train_enc shape ', x_train_enc.shape)

            if task_type == 'binary':
                # print('single tree prediction shape', single_tree_prediction(
                #     temp_trees[i], j, x_train, task_type=task_type).shape)
                x_train_enc[:, i * client_tree_num + j] = single_tree_prediction(
                    temp_trees[i], j, x_train, task_type=task_type
                )

                # print('x_train_enc[:, i * client_tree_num + j]  shape',
                #       x_train_enc[:, i * client_tree_num + j].shape)

            elif task_type == 'multiclass':
                #print('single_tree_prediction', single_tree_prediction(
                #temp_trees[i], j, x_train, task_type=task_type))
                # print('x_train_enc[:, i * client_tree_num + j]  shape',
                #       x_train_enc[:, i * client_tree_num + j*20 : i * client_tree_num + (j+1)*20].shape)

                x_train_enc[:, i * client_tree_num *20 + j*20 : i * client_tree_num*20 + (j+1)*20] = single_tree_prediction(
                temp_trees[i], j, x_train, task_type=task_type)


    x_train_enc32: Any = np.float32(x_train_enc)
    y_train32: Any = np.float32(y_train)

    x_train_enc32, y_train32 = torch.from_numpy(
        np.expand_dims(x_train_enc32, axis=1)  # type: ignore
    ), torch.from_numpy(
        np.expand_dims(y_train32, axis=-1)  # type: ignore
    )

    #print('x_train_enc[0]',x_train_enc[0])
    return x_train_enc32, y_train32


def tree_encoding_loader(
    dataloader: DataLoader,
    batch_size: int,
    client_trees: Union[
        Tuple[XGBClassifier, int],
        Tuple[XGBRegressor, int],
        List[Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]],
    ],
    client_tree_num: int,
    client_num: int,
    task_type: str = 'binary'
) -> DataLoader:
    #print('tastk type in tree_encoding_loader', task_type)
    encoding = tree_encoding(dataloader, client_trees, client_tree_num, client_num, task_type=task_type)
    if encoding is None:
        return None
    data, labels = encoding
    tree_dataset = TreeDataset(data, labels)
    return get_dataloader(tree_dataset, "tree", batch_size)

