# -*- coding:utf-8 -*-
__author__ = 'maomaochong'
import pandas as pd
import woe.config as config
import pickle
import time
import os

def eval_feature_detail(Info_Value_list,out_path=False):
    """
    format InfoValue list to Dataframe
    :param Info_Value_list: Instance list of Class InfoValue
    :param out_path:specify the Dataframe to csv file path ,default False
    :return:DataFrame about feature detail
    """
    rst = Info_Value_list
    format_rst = []

    for kk in range(0,len(rst)):
        print  rst[kk].var_name
        split_list = []
        if rst[kk].split_list != []:
            if not rst[kk].is_discrete:
                #deal with split_list
                split_list.append('(-INF,'+str(rst[kk].split_list[0])+']')
                for i in range(0,len(rst[kk].split_list)-1):
                    split_list.append('(' + str(rst[kk].split_list[i])+','+ str(rst[kk].split_list[i+1]) + ']')

                split_list.append('(' + str(rst[kk].split_list[len(rst[kk].split_list)-1]) + ',+INF)')
            else:
                split_list = rst[kk].split_list
        else:
            split_list.append('(-INF,+INF)')

        # merge into dataframe
        columns = ['var_name','split_list','sub_total_sample_num','positive_sample_num'
            ,'negative_sample_num','sub_total_num_percentage','positive_rate_in_sub_total'
            ,'woe_list','iv_list','iv']
        rowcnt = len(rst[kk].iv_list)
        if rowcnt < len(split_list):
            split_list = split_list[:rowcnt]

        var_name = [rst[kk].var_name] * rowcnt
        iv = [rst[kk].iv] * rowcnt
        iv_list = rst[kk].iv_list
        woe_list = rst[kk].woe_list
        a = pd.DataFrame({'var_name':var_name,'iv_list':iv_list,'woe_list':woe_list
                             ,'split_list':split_list,'iv':iv,'sub_total_sample_num':rst[kk].sub_total_sample_num
                             ,'positive_sample_num':rst[kk].positive_sample_num,'negative_sample_num':rst[kk].negative_sample_num
                             ,'sub_total_num_percentage':rst[kk].sub_total_num_percentage
                             ,'positive_rate_in_sub_total':rst[kk].positive_rate_in_sub_total
                             ,'negative_rate_in_sub_total':rst[kk].negative_rate_in_sub_total},columns=columns)
        format_rst.append(a)

    # merge dataframe list into one dataframe vertically
    cformat_rst = pd.concat(format_rst)

    if out_path:
        file_name = out_path if isinstance(out_path, str) else None
        cformat_rst.to_csv(file_name, index=False,encoding='utf-8')

    return cformat_rst



rst_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model\\gendata\\WOE_Rule\\cs_m1_pos_woe_rule_201705.pkl'
output = open(rst_path, 'rb')
rst = pickle.load(output)

output_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model\\features_detail\\cs_m1_pos_201705_features_detail.csv'

eval_feature_detail(rst,out_path=output_path)


import woe.eval as eval
output_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model\\features_detail\\cs_m1_pos_201705_features_summary.csv'
candidate_var_list = [
    'avg_days'
    ,'bptp_ratio'
    ,'city'
    ,'due_periods_ratio'
    ,'finish_periods_ratio'
    ,'intime_pay'
    ,'kptp_ratio'
    ,'most_contact_3m'
    ,'person_app_age'
    ,'person_sex'
    ,'recent_contact_day'
    ,'rej_count'
    ,'rpy_cn'
    ,'seq_delay_days'
    ,'state_sagroup'
    ,'tot_credit_amount'
    ,'value_balance_ratio'
    ,'value_income_ratio']

model_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model\\gendata\\LogisticRegression_Model\\cs_m1_pos_clf_201705.pkl'
output = open(model_path, 'rb')
clf = pickle.load(output)

i = 4
j = 4
dataset_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model\\gendata\\' + 'cs_m1_pos_woe_transed_rule_20170'+str(i+1) \
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


eval.eval_feature_summary(X_train,clf,rst,candidate_var_list,out_path=output_path)


# 日模型feature detail 中文编码问题
print('%s\tGET CONFIG' %(time.asctime(time.localtime(time.time()))))
config_path = r'E:\Code\Python_ML_Code\cs_model\config\config_cs_daily_model.csv'
cfg = config.config()
cfg.load_file(config_path)

feature_list = list(cfg.bin_var_list)
feature_list.extend(list(cfg.discrete_var_list))
rst = []
rst_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\WOE_Rule\\'
rst_list = os.listdir(rst_path)
rst_list = [rst_list[i].split(".")[0] for i in range(rst_list.__len__())]

# feature_list_todo = list(set(feature_list)^set(rst_list))
feature_rst_list = list(set(feature_list).intersection(set(rst_list)))

print('%s\tGET WOE RULE START' %(time.asctime(time.localtime(time.time()))))
for i in range(feature_rst_list.__len__()):
    var = feature_rst_list[i]
    rst_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\WOE_Rule\\' + var + '.pkl'
    output = open(rst_path, 'rb')
    civ = pickle.load(output)
    output.close()
    rst.append(civ[0])
print('%s\tGET WOE RULE END' %(time.asctime(time.localtime(time.time()))))

outfile_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\features_detail\\cs_m1_pos_model_daily_features_detail_utf8.csv'
feature_detail = eval_feature_detail(rst, outfile_path)