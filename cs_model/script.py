# -*- coding:utf-8 -*-
__author__ = 'maomaochong'
import pandas as pd
import numpy as np
import reportgen as rpt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge

dataset_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\m1_cs_201705_daily_new_sub_woe_transed_train.csv'
config_path = r'E:\Code\Python_ML_Code\cs_model\config\config_cs_daily_model.csv'

dataset_train = pd.read_csv(dataset_path)

cfg = pd.read_csv(config_path)
candidate_var_list = cfg[cfg['is_modelfeature'] == 1]['var_name']

b = [var for var in dataset_train.columns if sum(dataset_train[var].isnull()) == 0]
candidate_var_list = list(set(candidate_var_list).intersection(set(b)))

var_list_specfied = ['value_income_ratio'
    ,'avg_rollseq'
    ,'csfq'
    ,'bptp_ratio'
    ,'cert_4_inital'
    ,'ptp'
    ,'city'
    ,'person_sex'
    ,'over_due_value'
    ,'most_contact_3m'
    ,'recent_contact_day'
    ,'rej_count'
    ,'avg_days'
    ,'max_cpd'
    ,'finish_periods_ratio'
    ,'due_periods_ratio'
    ,'seq_delay_days']

candidate_var_list = list(set(candidate_var_list).intersection(set(var_list_specfied)))
print 'candidate_var_list:\n',candidate_var_list

print 'change dtypes:float64 to float32'
for var in candidate_var_list:
    dataset_train[var] = dataset_train[var].astype(np.float32)

dataset_train['trans_cs_cpd'] = np.log(dataset_train.raw_cs_cpd/31.0)- np.log(1-dataset_train.raw_cs_cpd/31.0)
candidate_var_list.append('trans_cs_cpd')

X_train = dataset_train[candidate_var_list]
y_train = dataset_train['target']

plt.figure()
n_alphas = 20
alphas = np.logspace(-8,1,num=n_alphas)
coefs = []
for a in alphas:
    ridge = Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X_train, y_train)
    coefs.append(ridge.coef_)
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
handles, labels = ax.get_legend_handles_labels()
plt.legend(labels=labels)
plt.show()

#密度函数和KL距离
kl_div_score1=rpt.utils.entropyc.kl_div(y_train_proba[y_train==0,0],y_train_proba[y_train==1,0])
kl_div_score2=rpt.utils.entropyc.kl_div(y_train_proba[y_train==1,0],y_train_proba[y_train==0,0])
kl_div_score=kl_div_score1+kl_div_score2
fig,ax=plt.subplots()
sns.distplot(y_train_proba[y_train==0,0],ax=ax,label='good')
sns.distplot(y_train_proba[y_train==1,0],ax=ax,label='bad')
ax.set_title('KL_DIV = %.5f'%kl_div_score)
ax.legend()
fig.show()