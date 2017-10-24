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


shop_info = pd.read_csv(r'E:\output\rawdata\ccf_first_round_shop_info.csv')

result = pd.merge(dataset, shop_info, how='left', on=['shop_id'])

from sklearn.model_selection import train_test_split
X = result[['longitude_x','latitude_x','longitude_y','latitude_y']]
y = result['shop_id']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=32)



shop_info = pd.read_csv(r'E:\output\rawdata\ccf_first_round_shop_info.csv')
evalset = pd.read_csv(r'E:\output\rawdata\evaluation_public.csv')

