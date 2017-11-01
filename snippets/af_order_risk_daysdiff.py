# -*- coding:utf-8 -*-
import pandas as pd
from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_leaf =0.05,max_depth =5)

dataset = pd.read_csv(r'E:\work\4_Strategy_analysis_and_optimization\fpd30_analysis\order_risk_daysdiff_tmp05.csv')

dataset_train = dataset[dataset['agr_fpd30'] == 1]
y = dataset_train['def_fpd30']
X = dataset_train[['avg_date_diff','order_cnt']]
clf = clf.fit(X, y)


import pydotplus
dot_data = tree.export_graphviz(clf, out_file=None,
                                filled=True, rounded=True,
                                feature_names=['avg_date_diff','order_cnt'],
                                special_characters=True,
                                proportion=True,
                                rotate=False,
                                max_depth=5
                                )
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf(r'E:\work\4_Strategy_analysis_and_optimization\fpd30_analysis\order_risk_daysdiff_fpd30.pdf')


clf = tree.DecisionTreeClassifier(min_samples_leaf =0.05)
dataset_train = dataset[dataset['agr_fpd4'] == 1]
y = dataset_train['def_fpd4']
X = dataset_train[['avg_date_diff','order_cnt']]
clf = clf.fit(X, y)

dot_data = tree.export_graphviz(clf, out_file=None,
                                filled=True, rounded=True,
                                feature_names=['avg_date_diff','order_cnt'],
                                special_characters=True,
                                proportion=True,
                                rotate=False
                                )
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf(r'E:\work\4_Strategy_analysis_and_optimization\fpd30_analysis\order_risk_daysdiff_fpd4.pdf')


