__author__ = 'maomaochong'
import pandas as pd
import woe.config as config
import woe.feature_process as fp
import woe.eval as eval
import woe.GridSearch as gs
import numpy as np
import pickle
import time

civ_list_path = r'E:\ScoreCard\cs_model\cs_m1_pos_model\gendata\WOE_Rule\cs_m1_pos_woe_rule_201704.pkl'

output = open(civ_list_path, 'rb')
civ_list = pickle.load(output)
output.close()

df_train = pd.read_csv(r'E:\ScoreCard\cs_model\cs_m1_pos_model\raw_data\m1_rsx_cs_unify_model_features_201704.csv')
df_validation = pd.read_csv(r'E:\ScoreCard\cs_model\cs_m1_pos_model\raw_data\m1_rsx_cs_unify_model_features_201705.csv')

config_path = r'E:\Code\Python_ML_Code\cs_model\config\config_cs_model.csv'

reload(config)
cfg = config.config()
cfg.load_file(config_path)

def fillna(dataset,bin_var_list,discrete_var_list,continuous_filler=-1,discrete_filler='missing'):
    """
    fill the null value in the dataframe inpalce
    :param dataset: input dataset ,pandas.DataFrame type
    :param bin_var_list:  continuous variables name list
    :param discrete_var_list: discretevvvv variables name list
    :param continuous_filler: the value to fill the null value in continuous variables
    :param discrete_filler: the value to fill the null value in discrete variables
    :return: null value,replace null value inplace
    """
    for var in [tmp for tmp in bin_var_list if tmp in list(dataset.columns)]:
        # fill null
        dataset.loc[dataset[var].isnull(), (var)] = continuous_filler

    for var in [tmp for tmp in discrete_var_list if tmp in list(dataset.columns)]:
        # fill null
        dataset.loc[dataset[var].isnull(), (var)] = discrete_filler

fillna(df_train,cfg.bin_var_list,cfg.discrete_var_list)
fillna(df_validation,cfg.bin_var_list,cfg.discrete_var_list)
fp.change_feature_dtype(df_train, cfg.variable_type)
fp.change_feature_dtype(df_validation, cfg.variable_type)

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

reload(eval)
eval.eval_feature_stability(civ_list, df_train, df_validation,candidate_var_list,out_path='E:\\feature_stability.csv')

import ScoreCardModel
