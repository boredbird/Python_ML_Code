# -*- coding:utf-8 -*-
from sklearn.neighbors import KNeighborsClassifier

X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

print(neigh.predict([[1.1]]))

print(neigh.predict_proba([[0.9]]))

import pandas as pd
import time
raw_data_path = r'E:\output\rawdata\ccf_first_round_user_shop_behavior.csv'

dataset = pd.read_csv(raw_data_path)

