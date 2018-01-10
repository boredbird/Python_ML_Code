# -*- coding:utf-8 -*-
__author__ = 'maomaochong'
import pandas as pd
import woe.config as config
import woe.feature_process as fp
import pickle
import time
import copy
import woe.eval as eval
import os

def process_woe_trans(cfg=None,rst=None,dataset=None,out_path=None):
    # fill null
    for var in [tmp for tmp in cfg.bin_var_list if tmp in list(dataset.columns)]:
        dataset.loc[dataset[var].isnull(), (var)] = -1

    for var in [tmp for tmp in cfg.discrete_var_list if tmp in list(dataset.columns)]:
        dataset.loc[dataset[var].isnull(), (var)] = 'missing'

    fp.change_feature_dtype(dataset, cfg.variable_type)

    for r in rst:
        dataset[r.var_name] = fp.woe_trans(dataset[r.var_name], r)

    dataset.to_csv(out_path)


def woe01_trans(dvar,civ):
    # replace the var value with the given woe value
    var = copy.deepcopy(dvar)
    if not civ.is_discrete:
        if civ.woe_list.__len__()>1:
            split_list = []
            split_list.append(float("-inf"))
            split_list.extend([i for i in civ.split_list])
            split_list.append(float("inf"))

            for i in range(civ.woe_list.__len__()):
                # var[(dvar > split_list[i]) & (dvar <= split_list[i+1])] = civ.woe_list[i]
                var[(dvar > split_list[i]) & (dvar <= split_list[i+1])] = i
        else:
            var[:] = 0
    else:
        split_map = {}
        for i in range(civ.split_list.__len__()):
            for j in range(civ.split_list[i].__len__()):
                # split_map[civ.split_list[i][j]] = civ.woe_list[i]
                split_map[civ.split_list[i][j]] = i

        var = var.map(split_map)

    return var

def process_woe01_trans(cfg=None,rst=None,dataset=None,out_path=None):
    # fill null
    for var in [tmp for tmp in cfg.bin_var_list if tmp in list(dataset.columns)]:
        dataset.loc[dataset[var].isnull(), (var)] = -1

    for var in [tmp for tmp in cfg.discrete_var_list if tmp in list(dataset.columns)]:
        dataset.loc[dataset[var].isnull(), (var)] = 'missing'

    fp.change_feature_dtype(dataset, cfg.variable_type)

    for r in rst:
        dataset[r.var_name] = woe01_trans(dataset[r.var_name], r)

    dataset.to_csv(out_path)


if __name__ == "__main__":
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

    # outfile_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\features_detail\\cs_m1_pos_model_daily_features_detail.csv'
    # feature_detail = eval.eval_feature_detail(rst, outfile_path)

    # do woe trans
    # for j in range(24):
    #     dataset_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\raw_data\\dataset_split_by_rows\\' \
    #                    + 'm1_rsx_cs_unify_model_features_201705_daily_new_' \
    #                    + str(j+1) + '.csv'
    #     dataset = pd.read_csv(dataset_path)
    #
    #     # out_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\dataset_split_by_rows\\' \
    #     #            + 'm1_rsx_cs_unify_model_features_201705_daily_' \
    #     #            + str(i+1) + '_woe_transed.csv'
    #     # print('%s\tDO WOE TRANSFORMATION\n%s' % (time.asctime(time.localtime(time.time())),out_path))
    #     # process_woe_trans(cfg,rst,dataset,out_path)
    #
    #     out_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\dataset_split_by_rows\\' \
    #                + 'm1_rsx_cs_unify_model_features_201705_daily_' \
    #                + str(j+1) + '_woe01_transed.csv'
    #     print('%s\tDO WOE TRANSFORMATION\n%s' % (time.asctime(time.localtime(time.time())),out_path))
    #     process_woe01_trans(cfg,rst,dataset,out_path)

    # 保留cs_cpd这个字段
    for j in range(24):
        dataset_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\raw_data\\dataset_split_by_rows\\' \
                       + 'm1_rsx_cs_unify_model_features_201706_daily_new_' \
                       + str(j+1) + '.csv'
        dataset = pd.read_csv(dataset_path)

        dataset = dataset.rename(columns={'cpd':'cs_cpd'}) # rename
        dataset['raw_cs_cpd'] = dataset['cs_cpd']
        # out_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\dataset_split_by_rows\\' \
        #            + 'm1_rsx_cs_unify_model_features_201705_daily_' \
        #            + str(i+1) + '_woe_transed.csv'
        # print('%s\tDO WOE TRANSFORMATION\n%s' % (time.asctime(time.localtime(time.time())),out_path))
        # process_woe_trans(cfg,rst,dataset,out_path)

        out_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\dataset_split_by_rows\\' \
                   + 'm1_rsx_cs_unify_model_features_201706_daily_' \
                   + str(j+1) + '_woe01_transed_with_cpd.csv'
        print('%s\tDO WOE TRANSFORMATION\n%s' % (time.asctime(time.localtime(time.time())),out_path))
        process_woe01_trans(cfg,rst,dataset,out_path)
