import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import *

from copy import copy
from sklearn.feature_selection import VarianceThreshold, SequentialFeatureSelector, SelectFromModel
from sklearn.ensemble import RandomForestClassifier


# filtering method
def filter_features(x, var_threshold=0.02, cor_threshold=0.8):
    # removing quansi-constant feature with threshold 0.02
    constant_filter = VarianceThreshold(var_threshold)
    constant_filter.fit(x)

    new_x = copy(x[x.columns[constant_filter.get_support()]])

    # removing correlated feature
    correlated_features = set()
    correlation_matrix = new_x.corr()

    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > cor_threshold:
                col_name = correlation_matrix.columns[i]
                correlated_features.add(col_name)

    new_x.drop(labels=correlated_features, axis=1, inplace=True)

    return new_x.values


# wrapper method
def wrap_feature(x, y, estimator, num_feature_keep, direction):
    sfs = SequentialFeatureSelector(estimator, direction=direction, n_features_to_select=int(num_feature_keep))

    new_x = sfs.fit_transform(x, y.values[:, 0])

    return new_x


# embedding method
def embed_feature(x, y, threshold=0.02):
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

    # select features
    sfm = SelectFromModel(rf, threshold=threshold)
    sfm.fit(x, y.values[:, 0])
    embed_x = sfm.transform(x)

    return embed_x
