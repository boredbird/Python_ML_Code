# -*- coding:utf-8 -*-
__author__ = 'maomaochong'
import numpy as np
import pandas as pd
import time
import pickle
import os
from woe.eval import  compute_ks
from woe.ftrl import *

print('########################训练集整体########################')
# 定义数据结构
model_ks_dict = {}
model_ks_dict['y_train'] = []
model_ks_dict['y_hat'] = []

enc_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\LogisticRegression_Model\\OneHotEncoder.pkl'
output = open(enc_path, 'rb')
enc = pickle.load(output)
output.close()
print 'LOAD OneHotEncoder PKL:\n',enc_path

pkl_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\LogisticRegression_Model\\'
recent_path = max(os.listdir(pkl_path))
recent_path = pkl_path + recent_path
output = open(recent_path, 'rb')
ftrl = pickle.load(output)
output.close()
print 'LOAD  FTRL MODEL FILE:\n',recent_path

config_path = 'E:\\Code\\Python_ML_Code\\cs_model\\config\\config_cs_daily_model.csv'
print('%s\tLOAD CONFIG FILE:\n%s' % (time.asctime(time.localtime(time.time())),config_path))
cfg = pd.read_csv(config_path)
candidate_var_list = cfg[cfg['is_modelfeature'] == 1]['var_name']

# 分cpd
for i in range(23):
    # dataset
    dataset_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\dataset_split_by_rows\\m1_rsx_cs_unify_model_features_201705_daily_' \
                   +str(i+1)+'_woe01_transed_with_cpd.csv'

    dataset = pd.read_csv(dataset_path)

    b = [var for var in dataset.columns if sum(dataset[var].isnull()) == 0]
    candidate_var_list = list(set(candidate_var_list).intersection(set(b)))

    X_train = dataset[candidate_var_list].values
    y_train = dataset['target'].values
    model_ks_dict['y_train'].extend(y_train)

    X_train_enc = enc.transform(X_train).toarray()
    y_hat = 1.0 / (1.0 + np.exp(X_train_enc.dot(ftrl.w)))
    model_ks_dict['y_hat'].extend(y_hat)

ks = compute_ks(np.array(model_ks_dict['y_hat']),np.array(model_ks_dict['y_train']))
sample_cnt = model_ks_dict['y_train'].__len__()
bad_sample_cnt = sum(model_ks_dict['y_train'])
bad_sample_rate = bad_sample_cnt*1.0/sample_cnt
print('Overall K-S Performance of The Model:\nsample_cnt:%s\tbad_sample_cnt:%s\tbad_sample_rate:%s\tks:%s'
      % (str(sample_cnt),str(bad_sample_cnt),str(bad_sample_rate),str(ks)))
"""
Overall K-S Performance of The Model:
sample_cnt:11500000	bad_sample_cnt:5380205	bad_sample_rate:0.467843913043	ks:0.522420624487
"""

print '########################同时期验证集整体########################'
pkl_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\LogisticRegression_Model\\'
recent_path = max(os.listdir(pkl_path))
recent_path = pkl_path + recent_path
output = open(recent_path, 'rb')
ftrl = pickle.load(output)
output.close()

validation_dataset_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\dataset_split_by_rows\\m1_rsx_cs_unify_model_features_201705_daily_24_woe01_transed.csv'
validation_dataset = pd.read_csv(validation_dataset_path)

config_path = 'E:\\Code\\Python_ML_Code\\cs_model\\config\\config_cs_daily_model.csv'
print('%s\tLOAD CONFIG FILE:\n%s' % (time.asctime(time.localtime(time.time())),config_path))
cfg = pd.read_csv(config_path)
candidate_var_list = cfg[cfg['is_modelfeature'] == 1]['var_name']

b = [var for var in validation_dataset.columns if sum(validation_dataset[var].isnull()) == 0]
candidate_var_list = list(set(candidate_var_list).intersection(set(b)))

X_train = validation_dataset[candidate_var_list].values
y_train = validation_dataset['target'].values

enc_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\LogisticRegression_Model\\OneHotEncoder.pkl'
output = open(enc_path, 'rb')
enc = pickle.load(output)
output.close()
print 'LOAD OneHotEncoder PKL:\n',enc_path

X_train_enc = enc.transform(X_train).toarray()

y_hat = 1.0 / (1.0 + np.exp(X_train_enc.dot(ftrl.w)))

ks = compute_ks(np.array(y_hat),np.array(y_train))
sample_cnt = y_train.__len__()
bad_sample_cnt = sum(y_train)
bad_sample_rate = bad_sample_cnt*1.0/sample_cnt
print('Overall K-S Performance of The Model:\nsample_cnt:%s\tbad_sample_cnt:%s\tbad_sample_rate:%s\tks:%s'
      % (str(sample_cnt),str(bad_sample_cnt),str(bad_sample_rate),str(ks)))
"""
Overall K-S Performance of The Model:
sample_cnt:432694	bad_sample_cnt:201996	bad_sample_rate:0.466833374163	ks:0.523643758392
"""

print '########################训练集分cpd########################'
# 定义数据结构
model_ks_dict = {}
for i in range(30):
    model_ks_dict[i+1] = {}
    model_ks_dict[i+1]['y_train'] = []
    model_ks_dict[i+1]['y_hat'] = []

enc_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\LogisticRegression_Model\\OneHotEncoder.pkl'
output = open(enc_path, 'rb')
enc = pickle.load(output)
output.close()
print 'LOAD OneHotEncoder PKL:\n',enc_path

pkl_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\LogisticRegression_Model\\'
recent_path = max(os.listdir(pkl_path))
recent_path = pkl_path + recent_path
output = open(recent_path, 'rb')
ftrl = pickle.load(output)
output.close()
print 'LOAD  FTRL MODEL FILE:\n',recent_path

config_path = 'E:\\Code\\Python_ML_Code\\cs_model\\config\\config_cs_daily_model.csv'
print('%s\tLOAD CONFIG FILE:\n%s' % (time.asctime(time.localtime(time.time())),config_path))
cfg = pd.read_csv(config_path)
candidate_var_list = cfg[cfg['is_modelfeature'] == 1]['var_name']

# 分cpd
for i in range(23):
    # dataset
    dataset_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\dataset_split_by_rows\\m1_rsx_cs_unify_model_features_201705_daily_'\
                   +str(i+1)+'_woe01_transed_with_cpd.csv'

    dataset = pd.read_csv(dataset_path)

    b = [var for var in dataset.columns if sum(dataset[var].isnull()) == 0]
    candidate_var_list = list(set(candidate_var_list).intersection(set(b)))
    # cpd
    for j in range(30):
        subset = dataset[dataset['raw_cs_cpd']==(j+1)]

        X_train = subset[candidate_var_list].values
        y_train = subset['target'].values
        model_ks_dict[j+1]['y_train'].extend(y_train)

        X_train_enc = enc.transform(X_train).toarray()
        y_hat = 1.0 / (1.0 + np.exp(X_train_enc.dot(ftrl.w)))
        model_ks_dict[j+1]['y_hat'].extend(y_hat)


for j in range(30):
    ks = compute_ks(np.array(model_ks_dict[j+1]['y_hat']),np.array(model_ks_dict[j+1]['y_train']))
    sample_cnt = model_ks_dict[j+1]['y_train'].__len__()
    bad_sample_cnt = sum(model_ks_dict[j+1]['y_train'])
    bad_sample_rate = bad_sample_cnt*1.0/sample_cnt
    print('cs_cpd:%s\tsample_cnt:%s\tbad_sample_cnt:%s\tbad_sample_rate:%s\tks:%s'
          % (str(j+1),str(sample_cnt),str(bad_sample_cnt),str(bad_sample_rate),str(ks)))

"""
########################训练集分cpd########################
LOAD OneHotEncoder PKL:
E:\ScoreCard\cs_model\cs_m1_pos_model_daily\gendata\LogisticRegression_Model\OneHotEncoder.pkl
LOAD  FTRL MODEL FILE:
E:\ScoreCard\cs_model\cs_m1_pos_model_daily\gendata\LogisticRegression_Model\ftrl_20180108145549.pkl
Mon Jan 08 16:27:07 2018	LOAD CONFIG FILE:
E:\Code\Python_ML_Code\cs_model\config\config_cs_daily_model.csv
cs_cpd:1	sample_cnt:1544290	bad_sample_cnt:169400	bad_sample_rate:0.10969442268	ks:0.465790682853
cs_cpd:2	sample_cnt:1035614	bad_sample_cnt:169784	bad_sample_rate:0.163945253734	ks:0.412441314507
cs_cpd:3	sample_cnt:864793	bad_sample_cnt:178716	bad_sample_rate:0.206657546951	ks:0.390694278618
cs_cpd:4	sample_cnt:689401	bad_sample_cnt:180067	bad_sample_rate:0.261193412832	ks:0.346797195582
cs_cpd:5	sample_cnt:583639	bad_sample_cnt:181368	bad_sample_rate:0.310753736471	ks:0.329668789409
cs_cpd:6	sample_cnt:508040	bad_sample_cnt:182306	bad_sample_rate:0.358841823478	ks:0.308476365608
cs_cpd:7	sample_cnt:452763	bad_sample_cnt:183002	bad_sample_rate:0.404189388267	ks:0.295754182509
cs_cpd:8	sample_cnt:408643	bad_sample_cnt:183036	bad_sample_rate:0.447911746928	ks:0.279998639723
cs_cpd:9	sample_cnt:372049	bad_sample_cnt:183174	bad_sample_rate:0.492338374784	ks:0.267601319817
cs_cpd:10	sample_cnt:350098	bad_sample_cnt:183564	bad_sample_rate:0.524321761335	ks:0.254157055044
cs_cpd:11	sample_cnt:329946	bad_sample_cnt:183994	bad_sample_rate:0.557648827384	ks:0.252696501595
cs_cpd:12	sample_cnt:312064	bad_sample_cnt:183478	bad_sample_rate:0.587949907711	ks:0.245991029066
cs_cpd:13	sample_cnt:296542	bad_sample_cnt:182999	bad_sample_rate:0.617109886627	ks:0.240103558613
cs_cpd:14	sample_cnt:283532	bad_sample_cnt:183186	bad_sample_rate:0.646085803366	ks:0.233604257737
cs_cpd:15	sample_cnt:271900	bad_sample_cnt:183077	bad_sample_rate:0.673324751747	ks:0.232111414548
cs_cpd:16	sample_cnt:260477	bad_sample_cnt:182002	bad_sample_rate:0.698725799207	ks:0.224748091174
cs_cpd:17	sample_cnt:250602	bad_sample_cnt:181074	bad_sample_rate:0.722556084947	ks:0.221502927672
cs_cpd:18	sample_cnt:242384	bad_sample_cnt:180577	bad_sample_rate:0.74500379563	ks:0.214225576535
cs_cpd:19	sample_cnt:234784	bad_sample_cnt:180091	bad_sample_rate:0.767049713779	ks:0.211959629058
cs_cpd:20	sample_cnt:227798	bad_sample_cnt:179548	bad_sample_rate:0.788189536344	ks:0.207954744851
cs_cpd:21	sample_cnt:221130	bad_sample_cnt:178756	bad_sample_rate:0.808375163931	ks:0.198013177541
cs_cpd:22	sample_cnt:215205	bad_sample_cnt:178214	bad_sample_rate:0.828112729723	ks:0.195324668932
cs_cpd:23	sample_cnt:209403	bad_sample_cnt:177472	bad_sample_rate:0.847514123484	ks:0.190101843539
cs_cpd:24	sample_cnt:204133	bad_sample_cnt:176839	bad_sample_rate:0.866293054038	ks:0.184310986383
cs_cpd:25	sample_cnt:198927	bad_sample_cnt:175967	bad_sample_rate:0.884580775863	ks:0.17299423114
cs_cpd:26	sample_cnt:193958	bad_sample_cnt:175110	bad_sample_rate:0.902824322792	ks:0.167320531678
cs_cpd:27	sample_cnt:192696	bad_sample_cnt:177475	bad_sample_rate:0.92101029601	ks:0.167360871515
cs_cpd:28	sample_cnt:189055	bad_sample_cnt:178198	bad_sample_rate:0.94257226733	ks:0.173206725701
cs_cpd:29	sample_cnt:181143	bad_sample_cnt:175916	bad_sample_rate:0.971144344523	ks:0.188480582457
cs_cpd:30	sample_cnt:174991	bad_sample_cnt:171815	bad_sample_rate:0.981850495168	ks:0.223976573347

"""
print '########################同时期验证集分cpd########################'
# 定义数据结构
model_ks_dict = {}
for i in range(30):
    model_ks_dict[i+1] = {}
    model_ks_dict[i+1]['y_train'] = []
    model_ks_dict[i+1]['y_hat'] = []

i = 23
dataset_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\dataset_split_by_rows\\m1_rsx_cs_unify_model_features_201705_daily_' \
               +str(i+1)+'_woe01_transed_with_cpd.csv'

dataset = pd.read_csv(dataset_path)

b = [var for var in dataset.columns if sum(dataset[var].isnull()) == 0]
candidate_var_list = list(set(candidate_var_list).intersection(set(b)))
# cpd
for j in range(30):
    subset = dataset[dataset['raw_cs_cpd']==(j+1)]

    X_train = subset[candidate_var_list].values
    y_train = subset['target'].values
    model_ks_dict[j+1]['y_train'].extend(y_train)

    X_train_enc = enc.transform(X_train).toarray()
    y_hat = 1.0 / (1.0 + np.exp(X_train_enc.dot(ftrl.w)))
    model_ks_dict[j+1]['y_hat'].extend(y_hat)


for j in range(30):
    ks = compute_ks(np.array(model_ks_dict[j+1]['y_hat']),np.array(model_ks_dict[j+1]['y_train']))
    sample_cnt = model_ks_dict[j+1]['y_train'].__len__()
    bad_sample_cnt = sum(model_ks_dict[j+1]['y_train'])
    bad_sample_rate = bad_sample_cnt*1.0/sample_cnt
    print('cs_cpd:%s\tsample_cnt:%s\tbad_sample_cnt:%s\tbad_sample_rate:%s\tks:%s'
          % (str(j+1),str(sample_cnt),str(bad_sample_cnt),str(bad_sample_rate),str(ks)))
"""
########################同时期验证集分cpd########################
cs_cpd:1	sample_cnt:58259	bad_sample_cnt:6317	bad_sample_rate:0.10842959886	ks:0.477593248011
cs_cpd:2	sample_cnt:39045	bad_sample_cnt:6435	bad_sample_rate:0.164809834806	ks:0.423362704963
cs_cpd:3	sample_cnt:32352	bad_sample_cnt:6630	bad_sample_rate:0.204933234421	ks:0.397885149287
cs_cpd:4	sample_cnt:25512	bad_sample_cnt:6601	bad_sample_rate:0.258740984635	ks:0.357387775271
cs_cpd:5	sample_cnt:21841	bad_sample_cnt:6768	bad_sample_rate:0.309875921432	ks:0.324101488595
cs_cpd:6	sample_cnt:19162	bad_sample_cnt:6916	bad_sample_rate:0.36092265943	ks:0.309958814233
cs_cpd:7	sample_cnt:16908	bad_sample_cnt:6740	bad_sample_rate:0.398627868465	ks:0.290451103946
cs_cpd:8	sample_cnt:15406	bad_sample_cnt:6813	bad_sample_rate:0.442230299883	ks:0.286218550188
cs_cpd:9	sample_cnt:13928	bad_sample_cnt:6857	bad_sample_rate:0.492317633544	ks:0.276557672593
cs_cpd:10	sample_cnt:13242	bad_sample_cnt:6879	bad_sample_rate:0.519483461713	ks:0.243317042439
cs_cpd:11	sample_cnt:12259	bad_sample_cnt:6798	bad_sample_rate:0.554531364712	ks:0.250188867661
cs_cpd:12	sample_cnt:11666	bad_sample_cnt:6827	bad_sample_rate:0.58520486885	ks:0.246726730501
cs_cpd:13	sample_cnt:11457	bad_sample_cnt:7111	bad_sample_rate:0.62066858689	ks:0.230191805013
cs_cpd:14	sample_cnt:10610	bad_sample_cnt:6794	bad_sample_rate:0.640339302545	ks:0.251797815806
cs_cpd:15	sample_cnt:10342	bad_sample_cnt:6969	bad_sample_rate:0.673854186811	ks:0.215295240193
cs_cpd:16	sample_cnt:10063	bad_sample_cnt:6999	bad_sample_rate:0.695518235119	ks:0.231526920854
cs_cpd:17	sample_cnt:9552	bad_sample_cnt:6917	bad_sample_rate:0.724141541039	ks:0.225760419218
cs_cpd:18	sample_cnt:9285	bad_sample_cnt:6931	bad_sample_rate:0.7464728056	ks:0.219203443287
cs_cpd:19	sample_cnt:8862	bad_sample_cnt:6825	bad_sample_rate:0.770142180095	ks:0.214259280239
cs_cpd:20	sample_cnt:8657	bad_sample_cnt:6835	bad_sample_rate:0.789534480767	ks:0.201930481468
cs_cpd:21	sample_cnt:8343	bad_sample_cnt:6719	bad_sample_rate:0.805345798873	ks:0.210058216645
cs_cpd:22	sample_cnt:8115	bad_sample_cnt:6694	bad_sample_rate:0.824892174985	ks:0.184056136904
cs_cpd:23	sample_cnt:7920	bad_sample_cnt:6694	bad_sample_rate:0.845202020202	ks:0.211135486431
cs_cpd:24	sample_cnt:7688	bad_sample_cnt:6662	bad_sample_rate:0.866545265349	ks:0.199134423336
cs_cpd:25	sample_cnt:7525	bad_sample_cnt:6641	bad_sample_rate:0.882524916944	ks:0.182260412997
cs_cpd:26	sample_cnt:7389	bad_sample_cnt:6668	bad_sample_rate:0.902422519962	ks:0.172400818033
cs_cpd:27	sample_cnt:7210	bad_sample_cnt:6638	bad_sample_rate:0.920665742025	ks:0.167445540299
cs_cpd:28	sample_cnt:6958	bad_sample_cnt:6527	bad_sample_rate:0.938056912906	ks:0.174934601479
cs_cpd:29	sample_cnt:6601	bad_sample_cnt:6389	bad_sample_rate:0.967883653992	ks:0.208292850034
cs_cpd:30	sample_cnt:6537	bad_sample_cnt:6402	bad_sample_rate:0.97934832492	ks:0.261772362803
"""




print('########################跨时期验证集整体########################')
enc_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\LogisticRegression_Model\\OneHotEncoder.pkl'
output = open(enc_path, 'rb')
enc = pickle.load(output)
output.close()
print 'LOAD OneHotEncoder PKL:\n',enc_path

pkl_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\LogisticRegression_Model\\'
recent_path = max(os.listdir(pkl_path))
recent_path = pkl_path + recent_path
output = open(recent_path, 'rb')
ftrl = pickle.load(output)
output.close()
print 'LOAD  FTRL MODEL FILE:\n',recent_path

config_path = 'E:\\Code\\Python_ML_Code\\cs_model\\config\\config_cs_daily_model.csv'
print('%s\tLOAD CONFIG FILE:\n%s' % (time.asctime(time.localtime(time.time())),config_path))
cfg = pd.read_csv(config_path)
candidate_var_list = cfg[cfg['is_modelfeature'] == 1]['var_name']

# 定义数据结构
model_ks_dict = {}
model_ks_dict['y_train'] = []
model_ks_dict['y_hat'] = []

# 分cpd
for i in range(24):
    # dataset
    dataset_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\dataset_split_by_rows\\m1_rsx_cs_unify_model_features_201706_daily_' \
                   +str(i+1)+'_woe01_transed_with_cpd.csv'

    dataset = pd.read_csv(dataset_path)

    cols_to_keep = candidate_var_list[:]
    cols_to_keep.append('target')
    sub_dataset = dataset[cols_to_keep]
    sub_dataset = sub_dataset.dropna(axis=0,how='any') #todo
    # b = [var for var in dataset.columns if sum(dataset[var].isnull()) == 0]
    # candidate_var_list = list(set(candidate_var_list).intersection(set(b)))

    X_train = sub_dataset[candidate_var_list].values
    y_train = sub_dataset['target'].values
    model_ks_dict['y_train'].extend(y_train)

    X_train_enc = enc.transform(X_train).toarray()
    y_hat = 1.0 / (1.0 + np.exp(X_train_enc.dot(ftrl.w)))
    model_ks_dict['y_hat'].extend(y_hat)

ks = compute_ks(np.array(model_ks_dict['y_hat']),np.array(model_ks_dict['y_train']))
sample_cnt = model_ks_dict['y_train'].__len__()
bad_sample_cnt = sum(model_ks_dict['y_train'])
bad_sample_rate = bad_sample_cnt*1.0/sample_cnt
print('Overall K-S Performance of The Model:\nsample_cnt:%s\tbad_sample_cnt:%s\tbad_sample_rate:%s\tks:%s'
      % (str(sample_cnt),str(bad_sample_cnt),str(bad_sample_rate),str(ks)))
"""
Overall K-S Performance of The Model:
sample_cnt:11745779	bad_sample_cnt:5713220	bad_sample_rate:0.486406223036	ks:0.528330319178
"""

"""
candidate_var_list = ['value_income_ratio',
 'person_app_age',
 'con5_due_times',
 'max_roll_seq',
 'csfq',
 'bptp_ratio',
 'cert_4_inital',
 'education',
 'ptp',
 'city',
 'cs_cpd',
 'rpy_cn',
 'person_sex',
 'over_due_value',
 'intime_pay',
 'avg_rollseq',
 'value_balance_ratio',
 'recent_contact_day',
 'app_count',
 'rej_count',
 'con1_due_times',
 'avg_days',
 'qq_length',
 'is_suixinhuan',
 'tot_credit_amount',
 'max_cpd',
 'con10_due_times',
 'kptp_ratio',
 'finish_periods_ratio',
 'max_overdue',
 'delay_days',
 'due_periods_ratio',
 'jobtime',
 'due_delay_ratio',
 'seq_delay_days',
 'payprinciple']
"""

print '########################跨时期验证集分cpd########################'
# 定义数据结构
model_ks_dict = {}
for i in range(30):
    model_ks_dict[i+1] = {}
    model_ks_dict[i+1]['y_train'] = []
    model_ks_dict[i+1]['y_hat'] = []

enc_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\LogisticRegression_Model\\OneHotEncoder.pkl'
output = open(enc_path, 'rb')
enc = pickle.load(output)
output.close()
print 'LOAD OneHotEncoder PKL:\n',enc_path

pkl_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\LogisticRegression_Model\\'
recent_path = max(os.listdir(pkl_path))
recent_path = pkl_path + recent_path
output = open(recent_path, 'rb')
ftrl = pickle.load(output)
output.close()
print 'LOAD  FTRL MODEL FILE:\n',recent_path

config_path = 'E:\\Code\\Python_ML_Code\\cs_model\\config\\config_cs_daily_model.csv'
print('%s\tLOAD CONFIG FILE:\n%s' % (time.asctime(time.localtime(time.time())),config_path))
cfg = pd.read_csv(config_path)
candidate_var_list = cfg[cfg['is_modelfeature'] == 1]['var_name']

# 分cpd
for i in range(24):
    # dataset
    dataset_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\dataset_split_by_rows\\m1_rsx_cs_unify_model_features_201706_daily_' \
                   +str(i+1)+'_woe01_transed_with_cpd.csv'

    dataset = pd.read_csv(dataset_path)

    cols_to_keep = candidate_var_list[:]
    cols_to_keep.append('target')
    cols_to_keep.append('raw_cs_cpd')
    dataset = dataset[cols_to_keep]
    dataset = dataset.dropna(axis=0,how='any') #todo

    # b = [var for var in dataset.columns if sum(dataset[var].isnull()) == 0]
    # candidate_var_list = list(set(candidate_var_list).intersection(set(b)))
    # cpd
    for j in range(30):
        subset = dataset[dataset['raw_cs_cpd']==(j+1)]

        X_train = subset[candidate_var_list].values
        y_train = subset['target'].values
        model_ks_dict[j+1]['y_train'].extend(y_train)

        X_train_enc = enc.transform(X_train).toarray()
        y_hat = 1.0 / (1.0 + np.exp(X_train_enc.dot(ftrl.w)))
        model_ks_dict[j+1]['y_hat'].extend(y_hat)


for j in range(30):
    ks = compute_ks(np.array(model_ks_dict[j+1]['y_hat']),np.array(model_ks_dict[j+1]['y_train']))
    sample_cnt = model_ks_dict[j+1]['y_train'].__len__()
    bad_sample_cnt = sum(model_ks_dict[j+1]['y_train'])
    bad_sample_rate = bad_sample_cnt*1.0/sample_cnt
    print('cs_cpd:%s\tsample_cnt:%s\tbad_sample_cnt:%s\tbad_sample_rate:%s\tks:%s'
          % (str(j+1),str(sample_cnt),str(bad_sample_cnt),str(bad_sample_rate),str(ks)))
"""
cs_cpd:1	sample_cnt:1635086	bad_sample_cnt:195612	bad_sample_rate:0.119634074293	ks:0.457345867779
cs_cpd:2	sample_cnt:1105284	bad_sample_cnt:196974	bad_sample_rate:0.178211210874	ks:0.407225267399
cs_cpd:3	sample_cnt:864573	bad_sample_cnt:191086	bad_sample_rate:0.221017774092	ks:0.381693015958
cs_cpd:4	sample_cnt:689295	bad_sample_cnt:193361	bad_sample_rate:0.280519951545	ks:0.34066677747
cs_cpd:5	sample_cnt:583499	bad_sample_cnt:194583	bad_sample_rate:0.333476149916	ks:0.322471097359
cs_cpd:6	sample_cnt:507636	bad_sample_cnt:195233	bad_sample_rate:0.38459250329	ks:0.303177672197
cs_cpd:7	sample_cnt:455035	bad_sample_cnt:195959	bad_sample_rate:0.430645994264	ks:0.290600758551
cs_cpd:8	sample_cnt:412787	bad_sample_cnt:196629	bad_sample_rate:0.476344943034	ks:0.271848545649
cs_cpd:9	sample_cnt:377241	bad_sample_cnt:196799	bad_sample_rate:0.521679774998	ks:0.262638634509
cs_cpd:10	sample_cnt:355008	bad_sample_cnt:196885	bad_sample_rate:0.554593135929	ks:0.247232308812
cs_cpd:11	sample_cnt:333749	bad_sample_cnt:196472	bad_sample_rate:0.588681913654	ks:0.241854713454
cs_cpd:12	sample_cnt:316958	bad_sample_cnt:196818	bad_sample_rate:0.620959243811	ks:0.237894046265
cs_cpd:13	sample_cnt:301498	bad_sample_cnt:196397	bad_sample_rate:0.651403989413	ks:0.232371038038
cs_cpd:14	sample_cnt:287792	bad_sample_cnt:195696	bad_sample_rate:0.679991104687	ks:0.231109138394
cs_cpd:15	sample_cnt:275795	bad_sample_cnt:194810	bad_sample_rate:0.706357983285	ks:0.22617342746
cs_cpd:16	sample_cnt:265965	bad_sample_cnt:194652	bad_sample_rate:0.731870734871	ks:0.2221962678
cs_cpd:17	sample_cnt:257407	bad_sample_cnt:194317	bad_sample_rate:0.754901770348	ks:0.220413429188
cs_cpd:18	sample_cnt:248747	bad_sample_cnt:193362	bad_sample_rate:0.77734404837	ks:0.216195145198
cs_cpd:19	sample_cnt:240919	bad_sample_cnt:192414	bad_sample_rate:0.79866677182	ks:0.213405134303
cs_cpd:20	sample_cnt:233431	bad_sample_cnt:190952	bad_sample_rate:0.818023313099	ks:0.210329649855
cs_cpd:21	sample_cnt:226836	bad_sample_cnt:189846	bad_sample_rate:0.836930645929	ks:0.206075112283
cs_cpd:22	sample_cnt:220863	bad_sample_cnt:188550	bad_sample_rate:0.853696635471	ks:0.204928124842
cs_cpd:23	sample_cnt:215237	bad_sample_cnt:187387	bad_sample_rate:0.870607748668	ks:0.204449104882
cs_cpd:24	sample_cnt:209559	bad_sample_cnt:185754	bad_sample_rate:0.886404306186	ks:0.20293599209
cs_cpd:25	sample_cnt:204134	bad_sample_cnt:184224	bad_sample_rate:0.902466027217	ks:0.194374755882
cs_cpd:26	sample_cnt:198817	bad_sample_cnt:182785	bad_sample_rate:0.919363032336	ks:0.194003340052
cs_cpd:27	sample_cnt:190272	bad_sample_cnt:178130	bad_sample_rate:0.93618609149	ks:0.18788938577
cs_cpd:28	sample_cnt:182486	bad_sample_cnt:174105	bad_sample_rate:0.954073189176	ks:0.197419172088
cs_cpd:29	sample_cnt:174358	bad_sample_cnt:170214	bad_sample_rate:0.97623280836	ks:0.214048215163
cs_cpd:30	sample_cnt:175512	bad_sample_cnt:173214	bad_sample_rate:0.986906878162	ks:0.219028298082
"""