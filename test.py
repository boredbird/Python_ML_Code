import pandas as pd
import imp
g = imp.load_source('group',r'E:\Code\Python_ML_Code\group_adj.py')

train=pd.read_csv('E:\\out_data_1130.csv')
train.columns = [col.split('.')[-1] for col in train.columns]
train_data_y=train['target_cpd']
train_data_x=train[train.columns[6:30]]

g = imp.load_source('group',r'E:\Code\Python_ML_Code\group_adj.py')
g.binContVar(train_data_x.al_m3_id_notbank_allnum,train_data_y,4,mmax=3,Acc=0.05)