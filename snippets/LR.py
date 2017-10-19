# -*- coding:utf-8 -*-
__author__ = 'chunhui.zhang'
import sys
sys.path.append(r'E:\Code\ScoreCard')
from config_params import  *
import load_data as ld
import change_feature_dtype as cfd
import data_binning as bin
import calculate_iv as ci
import woe_transformation as wt
from sklearn import linear_model


#加载数据
dataset =ld.load_from_csv(rawdata_path,['pos_model_var_tbl_train',])
dataset_train = dataset[0]
dataset_train.columns = [col.split('.')[-1] for col in dataset_train.columns]
dataset_len = len(dataset_train)

#千分位数字处理
#TODO 将带逗号的千分位数字变量作为参数
# dataset_train['max_credit_pos'] = dataset_train['max_credit_pos'].apply(cfd.remove_thousands_separator)
# dataset_train['max_instalment_pos'] = dataset_train['max_instalment_pos'].apply(cfd.remove_thousands_separator)
# dataset_train['pos_credit'] = dataset_train['pos_credit'].apply(cfd.remove_thousands_separator)
# dataset_train['pos_cur_banlance'] = dataset_train['pos_cur_banlance'].apply(cfd.remove_thousands_separator)
# dataset_train['sa_avg_apply_amount_pos'] = dataset_train['sa_avg_apply_amount_pos'].apply(cfd.remove_thousands_separator)

#改变数据类型
cfd.change_feature_dtype(dataset_train)

########分箱处理
bin_var_list = config[config['is_tobe_bin']==1]['var_name']
for var in bin_var_list:
    try:
        print var
        #计算最优分隔点
        split = bin.binning_data_split(dataset_train,var)
        # print split

        #合并小组别
        split = bin.check_point(dataset_train,var,split,dataset_len)
        # print split

        #跟进分隔点区间将连续值替换成分类变量
        dataset_train[var] = bin.c2d(dataset_train[var],split)
        # print set(dataset_train[var])
    except Exception, e:
        print 'ERROR DATA BIN:',var

#计算IV和WOE替换
candidate_var_list = config[config['is_candidate']==1]['var_name']
iv_list = pd.Series()
for var in candidate_var_list:
    #计算每个变量的IV值
    try:
        iv_list[var] = ci.calculate_iv(dataset_train,var)
        print var,' the iv value: ', iv_list[var]

        # WOE变换
        wt.woe_transformation(dataset_train,var)
    except Exception, e:
        print 'ERROR CALCULATE IV:',var

#####################
#计算IV和WOE替换
candidate_var_list = config[config['is_candidate']==1]['var_name']
iv_list = pd.Series()
for var in candidate_var_list:
    #计算每个变量的IV值
    try:
        # iv_list[var] = ci.calculate_iv(dataset_train,var)
        # print var,' the iv value: ', iv_list[var]

        # WOE变换
        wt.woe_transformation(dataset_train_woe,var)
    except Exception, e:
        print 'ERROR CALCULATE IV:',var


#训练LR模型
modelfeature_var_list = config[config['is_modelfeature']==1]['var_name']
dataset_train_woe = pd.read_csv(gen_data_path + gendata_file_name + '.csv')


clf = linear_model.LogisticRegression(C=1e5)
X = dataset_train[modelfeature_var_list]
y = dataset_train['target']
clf.fit(X, y)

from sklearn.cross_validation import train_test_split, cross_val_score
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=0)

for var in X.columns:
    print var
    print set(X[var])

clf.coef_
clf.intercept_
clf.score(X,y)
clf.predict(X.iloc[:5,])
y

#结果矩阵
rst = pd.DataFrame({'pre':clf.predict(X),'act':y})
b = rst.groupby(['pre', 'act'])['act'].count()
print b

#预测概率值
probas_ = clf.predict_log_proba(X)
#计算auc
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y,probas_[:, 1])
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
mean_tpr += interp(mean_fpr, fpr, tpr)
mean_tpr[0] = 0.0
roc_auc = auc(fpr, tpr)
print roc_auc
#plot ROC
import matplotlib.pyplot as plt
i = 0
plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

#计算KS值
max(tpr - fpr)

