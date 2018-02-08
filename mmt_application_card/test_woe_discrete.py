# -*- coding:utf-8 -*-
__author__ = 'maomaochong'
import pandas as pd
import woe.feature_process as fp
import woe.config as config

dataset_train_path1 = r'E:\work_file\mmt_application_card\raw_data\mmt_application_model_feature_ftrain1.csv'
config_path = r'E:\work_file\mmt_application_card\config\config_mmt_application_model.csv'

dataset = pd.read_csv(dataset_train_path1)
var = 'data_status'
dataset.loc[dataset[var].isnull(), (var)] = 'missing'
cfg = config.config()
cfg.load_file(config_path,dataset_train_path1)

print 'cfg.global_bt',cfg.global_bt
print 'cfg.global_gt',cfg.global_gt
print 'cfg.min_sample',cfg.min_sample
# rst = fp.proc_woe_discrete(dataset,var,cfg.global_bt,cfg.global_gt,cfg.min_sample,alpha=0.05)

var = 'pos_sales_commission'
fp.proc_woe_continuous(dataset,var,cfg.global_bt,cfg.global_gt,cfg.min_sample,alpha=0.05)

var = 'pos_dd_fail_cnt'
fp.proc_woe_continuous(dataset,var,cfg.global_bt,cfg.global_gt,cfg.min_sample,alpha=0.05)
