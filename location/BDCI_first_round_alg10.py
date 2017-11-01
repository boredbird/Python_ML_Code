#-*- coding:utf-8 -*-

"""
"""

import pandas as pd
from collections import defaultdict

user_shop_hehavior = pd.read_csv(r'E:\output\rawdata\ccf_first_round_user_shop_behavior.csv')
evalution = pd.read_csv(r'E:\output\rawdata\evaluation_public.csv')

#让WIFI关联商铺

#构造规则
wifi_to_shops = defaultdict(lambda : defaultdict(lambda :0))
for line in user_shop_hehavior.values:
    wifi = sorted([wifi.split('|') for wifi in line[5].split(';')],key=lambda x:int(x[1]),reverse=True)[0]
    wifi_to_shops[wifi[0]][line[1]] = wifi_to_shops[wifi[0]][line[1]] + 1

right_count = 0
for line in user_shop_hehavior.values:
    wifi = sorted([wifi.split('|') for wifi in line[5].split(';')],key=lambda x:int(x[1]),reverse=True)[0]
    counter = defaultdict(lambda : 0)
    for k,v in wifi_to_shops[wifi[0]].items():
        counter[k] += v
    pred_one = sorted(counter.items(),key=lambda x:x[1],reverse=True)[0][0]
    if pred_one == line[1]:
        right_count += 1
print('acc:',right_count*1.0/len(user_shop_hehavior)) #线下验证
# ('acc:', 0.8704691941670365)

#预测
preds = []
for line in evalution.values:
    index = 0
    while True:
        try:
            if index==5:
                pred_one = None
                break
            wifi = sorted([wifi.split('|') for wifi in line[6].split(';')],key=lambda x:int(x[1]),reverse=True)[index]
            counter = defaultdict(lambda : 0)
            for k,v in wifi_to_shops[wifi[0]].items():
                counter[k] += v
            pred_one = sorted(counter.items(),key=lambda x:x[1],reverse=True)[0][0]
            break
        except:
            index+=1
    preds.append(pred_one)

result = pd.DataFrame({'row_id':evalution.row_id,'shop_id':preds})
result.fillna('s_666').to_csv(r'E:\output\submit\BDCI_first_round_alg10_submit01.csv.csv',index=None) #随便填的 这里还能提高不少
# score:0.8347



"""
分mall查看准确率
"""
# 线下分mall验证
shop_info = pd.read_csv(r'E:\output\rawdata\ccf_first_round_shop_info.csv')
shop_info.index = shop_info['shop_id']
user_shop_hehavior['mall_id'] = shop_info.loc[user_shop_hehavior['shop_id'] ,]['mall_id'].tolist()
user_shop_hehavior.index = user_shop_hehavior['mall_id']

mall_right_agg = defaultdict(lambda :0)
mall_cnt_agg = defaultdict(lambda :0)
for line in user_shop_hehavior.values:
    try:
        wifi = sorted([wifi.split('|') for wifi in line[5].split(';')], key=lambda x: int(x[1]), reverse=True)[0]
        counter = defaultdict(lambda : 0)
        for k,v in wifi_to_shops[wifi[0]].items():
            counter[k] += v
        pred_one = sorted(counter.items(),key=lambda x:x[1],reverse=True)[0][0]
        mall_cnt_agg[line[6]] += 1
        if pred_one == line[1]:
            mall_right_agg[line[6]] += 1
    except:
        continue

mall_acc = defaultdict(lambda :0)
for var in mall_cnt_agg.items():
    mall_acc[var[0]] = mall_right_agg[var[0]]*1.0/mall_cnt_agg[var[0]]

df_mall_acc = pd.DataFrame(mall_acc.items(),columns=['mall_id','acc'])
df_mall_acc.to_csv(r'E:\output\gendata\df_mall_acc_alg10.csv.csv',index=None)
