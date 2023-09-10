import pandas as pd
from sklearn.utils import resample
from sklearn.manifold import TSNE, MDS
from imblearn.over_sampling import RandomOverSampler, SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

def random_oversample(df, seed=1):
  X = df.loc[:, df.columns != 'label']
  y = df['label']
  sm = RandomOverSampler(random_state=seed)
  X_res, y_res = sm.fit_resample(X, y)
  res = pd.concat([X_res, y_res], axis=1)
  res.columns = df.columns
  return res

def oversample(df, n, seed=1):
  grouped = df.groupby('label')
  oversampled = []
  for label in grouped.groups.keys():
    oversampled.append(resample(
      df.loc[grouped.groups[label]],
      replace=True,
      n_samples=n,
      random_state=seed
    ))
  return pd.concat(oversampled)

def smote_oversample(df, k=8, seed=1):
  grouped = df.groupby('label')
  labels_rd = grouped.size()[grouped.size() <= k].index
  df_rd = df.loc[df['label'].isin(labels_rd)]
  res_rd = oversample(df_rd, seed=seed, n=max(grouped.size()))
  df_sm = df.loc[~df['label'].isin(labels_rd)]
  X = df_sm.loc[:, df.columns != 'label']
  y = df_sm.label
  sm = SMOTE(sampling_strategy='auto', k_neighbors=k, random_state=seed)
  X_res, y_res = sm.fit_resample(X, y)
  res_sm = pd.concat([X_res, y_res], axis=1)
  res_sm.columns = df.columns
  return pd.concat([res_rd, res_sm])

def visualise_samples(df, type='mds', seed=1):
  targets = df['label']
  if type == 'tsne':
    embedded = TSNE(n_components = 2, init='random', learning_rate='auto', perplexity = 50, n_iter = 1000, random_state=seed).fit_transform(df.loc[:, df.columns != 'label'].to_numpy())
  elif type == 'mds':
    embedded = MDS(random_state=seed).fit_transform(df.loc[:, df.columns != 'label'].to_numpy())
  plt.figure(figsize=(20,10))
  sns.scatterplot(
    x=embedded[:, 0],
    y=embedded[:, 1],
    hue=targets,
    # palette=sns.color_palette("deep"),
    legend="full"
  )
