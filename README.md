# Credit Risk Forecasting

This project aims to assess the effectiveness of federated learning (FL) in credit risk assessment. The main objectives are:

(1) To explore the integration of neural network techniques such as MLP and LSTM and tree ensemble techniques such as XGBoost. The focus is to fill the current research gap in FL for credit rating prediction by expanding the scope beyond the prevailing focus on logistic regression models. This work aims to explore the potential
benefits and applications of these models within a FL framework for credit risk forecasting

(2) To evaluate the effectiveness of FL models relative to local models and to investigate the impact of data imbalance on model performance, this research will employ MSE and AUC. This evaluation will particularly focus on both dominant clients with substantial datasets and non-dominant clients with limited datasets, with the aim of measuring performance improvement.

(3) To assess the robustness of FL in scalability and non-IID scenarios, this research will compare FL models against centralised models, utilising the MSE and AUC metrics.
These experiments will include a range of client numbers, from 2 to 10, and will consider both balanced and imbalanced data distributions

## Datasets

There datasets are stored in `dataset/`, `dataset2/` and `dataset3/`
Due to the size of the original dataset, Dataset 2 is not uploaded to the repository. 
- Dataset 1 is from https://doi.org/10.1016/j.eswa.2020.113925
- Dataset 2 is from https://www.kaggle.com/competitions/home-credit-default-risk
- Dataset 3 is from https://www.kaggle.com/competitions/GiveMeSomeCredit

## Main files

- `main1_lstm_central.ipynb`, `main1_lstm_federated.ipynb` and other main files contain the main functions for either centralised/local models or federated models
- main1, main2 and main3 represent the model is designed for Dataset 1, Dataset 2 and Dataset 3
- lstm, mlp and xgb represent which model is used in this file
- central or federated represent it is a centralised/local model or federated model 

## Other files/folders
- `resample.py` contains helper functions for different resampling functions and visualisations for the first dataset
- `fl_simu.py` contains the dataset partitioning strategies and the federated averaging algorithm for each model and dataset
- `fl_server.py` contains the FL server designed for XGBoost model
- `trees.py` contains helper functions for FL XGBoost model  
- `net_archs.py` contains the neural network architectures used in all datasets
- `clients/fl_client` contains the abstract class `FLClient` for the FL clients
- `clients/xgb_client` and `clients/xgb_multi_client` contain FL XGBoost clients for binary and multiclass classicications
- `clients/*`, the remaining clients are subclasses of `FLClient` that detail the train and test method for used in each network architecture

-  `eda.ipynb` and `eda2.ipynb` contain exploratory data analysis on Dataset 1 and Dataset 2
-  `model_checkpoints/` and LSTM in `lstm_model_checkpoints/` are used for saving FL models during training
