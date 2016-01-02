import functools
import pandas as pd
import numpy as np
import sklearn.linear_model
import sklearn.ensemble
import sklearn.metrics
from sklearn.cluster import KMeans

import scoring


class PCA(object):
    def __init__(self, n_princ_comp):
        self.n_princ_comp = n_princ_comp

    def fit(self, A):
        M = A - np.mean(A, axis=0)  # subtract the mean (along columns)
        self.eig_vals, self.eig_vecs = np.linalg.eig(np.cov(M.T))
        sorted_idx = np.argsort(self.eig_vals)
        self.eig_vecs_kept = self.eig_vecs[:, sorted_idx[-self.n_princ_comp:]]

    def transform(self, A):
        return A.dot(self.eig_vecs_kept)


def clean_dataset(df, quantile=0.95):
    means = df[scoring.float_cols].mean()
    stds = df[scoring.float_cols].std()
    zscores = (df[scoring.float_cols] - means).abs() / stds
    rowmax_zscores = zscores.max(axis=1)
    thresh = rowmax_zscores.quantile(quantile)
    idx = rowmax_zscores > thresh
    return df.drop(df.index[idx])


def balance_dataset(df):
    dfs = []
    for c, dfc in df.groupby('c1'):
        target = dfc['target'].values
        ones = np.sum(target == 1)
        zeros = np.sum(target == 0)
        if ones != zeros:
            if ones > zeros:
                drop_idx = np.where(target == 1)[0][:ones - zeros]
            else:
                drop_idx = np.where(target == 0)[0][:zeros - ones]
            dfc = dfc.drop(dfc.index[drop_idx])
        dfs.append(dfc)
    return pd.concat(dfs)


class DummyCoder(object):
    def fit(self, series):
        self.unique_vals = set(series.unique())
        self.unique_vals.pop()

    def encode(self, series):
        df_encoded = pd.DataFrame()
        for i, v in enumerate(self.unique_vals):
            df_encoded['d{}'.format(i)] = series.apply(lambda x: 1 if x == v else 0)
        return df_encoded.values


class MixedModelKmeans(object):
    def __init__(self, ycol='target', xcols=scoring.float_cols, categorical_col='c1',
                 base_estimator=functools.partial(sklearn.linear_model.ElasticNetCV, l1_ratio=[.1, .5, .7, .9, .95, .99, 1]),
                 n_princ_comp=8,
                 n_clusters=2):
        self.ycol = ycol
        self.xcols = xcols
        self.categorical_col = categorical_col
        self.n_clusters = n_clusters
        self.base_estimator = base_estimator
        self.n_princ_comp = n_princ_comp

    def fit(self, df):
        self.dummycoder = DummyCoder()
        self.dummycoder.fit(df[self.categorical_col])
        self.kmodel = KMeans(n_clusters=self.n_clusters, random_state=1332)

        clusters = self.kmodel.fit_predict(df[self.xcols])
        self.cluster_to_model = {}
        # clusters = df['c1'].values
        for cluster in set(clusters):
            dfc = df.iloc[clusters == cluster]
            pca = PCA(self.n_princ_comp)
            pca.fit(dfc[self.xcols])
            x = np.hstack((pca.transform(dfc[self.xcols]), self.dummycoder.encode(dfc[self.categorical_col])))
            y = dfc[self.ycol]
            model = self.base_estimator()
            model.fit(x, y)
            self.cluster_to_model[cluster] = model, pca

    def predict(self, df):
        clusters = self.kmodel.predict(df[self.xcols])
        res = np.zeros(len(df))
        clusters = df['c1'].values
        for c in set(clusters):
            model, pca = self.cluster_to_model[c]
            idx = clusters == c
            x = pca.transform(df[self.xcols][idx])
            dc = self.dummycoder.encode(df[self.categorical_col][idx])
            x = np.hstack((x, dc))
            res[idx] = model.predict(x)
        return res

    def score(self, df):
        yhat = self.predict(df)
        return sklearn.metrics.roc_auc_score(df[self.ycol], yhat)
