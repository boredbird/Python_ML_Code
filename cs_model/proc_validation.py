# -*- coding:utf-8 -*-
__author__ = 'maomaochong'
import woe.feature_process as fp
import woe.config as config
import pickle
from woe.eval import  compute_ks

def process_woe_trans(in_data_path=None,rst_path=None,out_path=None,config_path=None):
    data_path = in_data_path
    cfg = config.config()
    cfg.load_file(config_path, data_path)

    for var in [tmp for tmp in cfg.bin_var_list if tmp in list(cfg.dataset_train.columns)]:
        # fill null
        cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = -1

    for var in [tmp for tmp in cfg.discrete_var_list if tmp in list(cfg.dataset_train.columns)]:
        # fill null
        cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = 'missing'

    fp.change_feature_dtype(cfg.dataset_train, cfg.variable_type)

    output = open(rst_path, 'rb')
    rst = pickle.load(output)
    output.close()

    # Training dataset Woe Transformation
    for r in rst:
        cfg.dataset_train[r.var_name] = fp.woe_trans(cfg.dataset_train[r.var_name], r)

    cfg.dataset_train.to_csv(out_path)


def proc_validattion(dataset_path,config_path,model_path):
    print '####PROC VALIDATION#####'
    print 'dataset_path:\n',dataset_path
    print 'config_path:\n',config_path
    print 'model_path:\n',model_path
    #fillna
    cfg = config.config()
    cfg.load_file(config_path, dataset_path)

    for var in [tmp for tmp in cfg.bin_var_list if tmp in list(cfg.dataset_train.columns)]:
        # fill null
        cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = 0

    for var in [tmp for tmp in cfg.discrete_var_list if tmp in list(cfg.dataset_train.columns)]:
        # fill null
        cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = 0

    output = open(model_path, 'rb')
    clf_model = pickle.load(output)
    output.close()

    clf = clf_model['clf']
    X_test = cfg.dataset_train[clf_model['features_list']]
    y_test = cfg.dataset_train['target']

    y_hat = clf.predict_proba(X_test)[:,1]
    ks = compute_ks(y_hat,y_test)
    print 'global_bt:',cfg.global_bt
    print 'global_gt:', cfg.global_gt
    print 'ks:',ks
    return ks


if __name__ == '__main__':
    dataset_path = r'E:\ScoreCard\cs_model\cs_m1_pos_model\raw_data\m1_rsx_cs_unify_model_features_201712.csv'
    rst_path = r'E:\ScoreCard\cs_model\cs_m1_pos_model\gendata\WOE_Rule\cs_m1_pos_woe_rule_201711.pkl'
    outfile_path = r'E:\ScoreCard\cs_model\cs_m1_pos_model\gendata\cs_m1_pos_woe_transed_rule_201711_features_201712.csv'
    config_path = r'E:\Code\Python_ML_Code\cs_model\config\config_cs_model.csv'
    process_woe_trans(dataset_path,rst_path,outfile_path,config_path)

    proc_validattion(dataset_path=outfile_path
                     ,config_path=config_path
                     ,model_path=r'E:\ScoreCard\cs_model\cs_m1_pos_model\gendata\LogisticRegression_Model\cs_m1_pos_clf_201711.pkl')
