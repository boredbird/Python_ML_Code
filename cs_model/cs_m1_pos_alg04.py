# -*- coding:utf-8 -*-
__author__ = 'maomaochong'
import woe.GridSearch as gs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from datetime import datetime
from woe.eval import  compute_ks
import time
import pickle


def plot_ks(proba,target,axistype='pct',out_path=False):
    """
    plot k-s figure
    :param proba: 1-d array,prediction probability values
    :param target: 1-d array,the list of actual target value
    :param axistype: specify x axis :'axistype' must be either 'pct' (sample percent) or 'proba' (prediction probability)
    :param out_path: specify the file path to store ks plot figure,default False
    :return: DataFrame, figure summary
    """
    assert axistype in ['pct','proba'] , "KS Plot TypeError: Attribute 'axistype' must be either 'pct' or 'proba' !"

    a = pd.DataFrame(np.array([proba,target]).T,columns=['proba','target'])
    a.sort_values(by='proba',ascending=False,inplace=True)
    a['sum_Times']=a['target'].cumsum()
    total_1 = a['target'].sum()
    total_0 = len(a) - a['target'].sum()

    a['temp'] = 1
    a['Times']=a['temp'].cumsum()
    a['cdf1'] = a['sum_Times']/total_1
    a['cdf0'] = (a['Times'] - a['sum_Times'])/total_0
    a['ks'] = a['cdf1'] - a['cdf0']
    a['percent'] = a['Times']*1.0/len(a)

    idx = np.argmax(a['ks'])
    # print a.loc[idx]

    if axistype == 'pct':
        '''
        KS曲线,横轴为按照输出的概率值排序后的观察样本比例
        '''
        plt.figure()
        plt.plot(a['percent'],a['cdf1'], label="CDF_positive")
        plt.plot(a['percent'],a['cdf0'],label="CDF_negative")
        plt.plot(a['percent'],a['ks'],label="K-S")

        sx = np.linspace(0,1,10)
        sy = sx
        plt.plot(sx,sy,linestyle='--',color='darkgrey',linewidth=1.2)

        plt.legend()
        plt.grid(True)
        ymin, ymax = plt.ylim()
        plt.xlabel('Sample percent')
        plt.ylabel('Cumulative probability')
        plt.title('Model Evaluation Index K-S')
        plt.axis('tight')

        # 虚线
        t = a.loc[idx]['percent']
        yb = round(a.loc[idx]['cdf1'],4)
        yg = round(a.loc[idx]['cdf0'],4)

        plt.plot([t,t],[yb,yg], color ='red', linewidth=1.4, linestyle="--")
        plt.scatter([t,],[yb,], 20, color ='dodgerblue')
        plt.annotate(r'$recall_p=%s$' % round(a.loc[idx]['cdf1'],4), xy=(t, yb), xycoords='data', xytext=(+10, -5),
                     textcoords='offset points', fontsize=8,
                     arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.1"))

        plt.scatter([t,],[yg,], 20, color ='darkorange')
        plt.annotate(r'$recall_n=%s$' % round(a.loc[idx]['cdf0'],4), xy=(t, yg), xycoords='data', xytext=(+10, -10),
                     textcoords='offset points', fontsize=8,
                     arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.1"))
        # K-S曲线峰值
        plt.scatter([t,],[a.loc[idx]['ks'],], 20, color ='limegreen')
        plt.annotate(r'$ks=%s,p=%s$' % (round(a.loc[idx]['ks'],4)
                                        ,round(a.loc[idx]['proba'],4))
                     , xy=(a.loc[idx]['percent'], a.loc[idx]['ks'])
                     , xycoords='data'
                     , xytext=(+15, -15),
                     textcoords='offset points'
                     , fontsize=8
                     ,arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.1"))
        plt.annotate(r'$percent=%s,cnt=%s$' % (round(a.loc[idx]['percent'],4)
                                               ,round(a.loc[idx]['Times'],0))
                     , xy=(a.loc[idx]['percent'], a.loc[idx]['ks'])
                     , xycoords='data'
                     , xytext=(+25, -25),
                     textcoords='offset points'
                     , fontsize=8
                     )

    else:
        '''
        改变横轴,横轴为模型输出的概率值
        '''
        plt.figure()
        plt.grid(True)
        plt.plot(1-a['proba'],a['cdf1'], label="CDF_bad")
        plt.plot(1-a['proba'],a['cdf0'],label="CDF_good")
        plt.plot(1-a['proba'],a['ks'],label="ks")

        plt.legend()
        ymin, ymax = plt.ylim()
        plt.xlabel('1-[Predicted probability]')
        plt.ylabel('Cumulative probability')
        plt.title('Model Evaluation Index K-S')
        plt.axis('tight')
        plt.show()
        # 虚线
        t = 1 - a.loc[idx]['proba']
        yb = round(a.loc[idx]['cdf1'],4)
        yg = round(a.loc[idx]['cdf0'],4)

        plt.plot([t,t],[yb,yg], color ='red', linewidth=1.4, linestyle="--")
        plt.scatter([t,],[yb,], 20, color ='dodgerblue')
        plt.annotate(r'$recall_p=%s$' % round(a.loc[idx]['cdf1'],4), xy=(t, yb), xycoords='data', xytext=(+10, -5),
                     textcoords='offset points', fontsize=8,
                     arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.1"))

        plt.scatter([t,],[yg,], 20, color ='darkorange')
        plt.annotate(r'$recall_n=%s$' % round(a.loc[idx]['cdf0'],4), xy=(t, yg), xycoords='data', xytext=(+10, -10),
                     textcoords='offset points', fontsize=8,
                     arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.1"))
        # K-S曲线峰值
        plt.scatter([t,],[a.loc[idx]['ks'],], 20, color ='limegreen')
        plt.annotate(r'$ks=%s,p=%s$' % (round(a.loc[idx]['ks'],4)
                                        ,round(a.loc[idx]['proba'],4))
                     , xy=(t, a.loc[idx]['ks'])
                     , xycoords='data'
                     , xytext=(+15, -15),
                     textcoords='offset points'
                     , fontsize=8
                     ,arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.1"))
        plt.annotate(r'$percent=%s,cnt=%s$' % (round(a.loc[idx]['percent'],4)
                                               ,round(a.loc[idx]['Times'],0))
                     , xy=(t, a.loc[idx]['ks'])
                     , xycoords='data'
                     , xytext=(+25, -25),
                     textcoords='offset points'
                     , fontsize=8
                     )

    if out_path:
        file_name = out_path if isinstance(out_path, str) else None
        plt.savefig(file_name)
    else:
        plt.show()

    return a.loc[idx]

clf_path_list = [
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_clf_201701.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_clf_201702.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_clf_201703.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_clf_201704.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_clf_201705.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_clf_201706.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_clf_201707.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_clf_201708.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_clf_201709.pkl'
]

i = 4
j = 7
output = open(clf_path_list[i], 'rb')
clf = pickle.load(output)
output.close()

dataset_path = 'E:\\ScoreCard\\cs_model\\gendata\\' + 'cs_m1_pos_woe_transed_rule_20170'+str(i+1) \
               +'_features_20170'+str(j+1)+'.csv'
print dataset_path
dataset_train = pd.read_csv(dataset_path)

X_train = dataset_train
X_train = X_train[clf['features_name']]
y_train = dataset_train['target']

print 'Checking features dtypes:'
for var in clf['features_name']:
    # fill null
    X_train.loc[X_train[var].isnull(), (var)] = 0

proba = clf['classifier'].predict_proba(X_train)[:,1]
plot_ks(proba,y_train,out_path='E:\\201705.png')
