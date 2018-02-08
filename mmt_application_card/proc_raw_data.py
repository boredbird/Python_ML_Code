# -*- coding:utf-8 -*-
__author__ = 'maomaochong'
import pandas as pd

"""
更新data_status字段
"""
def update_data_status():
    dataset_path = r'E:\work_file\mmt_application_card\raw_data\mmt_application_model_feature_f.csv'
    pengyuan_path = r'E:\work_file\mmt_application_card\raw_data\pengyuan_edu.csv'

    dataset = pd.read_csv(dataset_path)
    pengyuan = pd.read_csv(pengyuan_path)

    list_pengyuan_application_no = list(pengyuan.application_no)

    new_data_status = []
    var = 'data_status'
    print sum(dataset['data_status'].isnull())
    for i in range(dataset['data_status'].__len__()):
        # if dataset['data_status'].isnull()[i] and dataset['application_no'][i] in list(pengyuan.application_no):
        if dataset['data_status'].isnull()[i] and dataset['application_no'][i] in list_pengyuan_application_no:
            new_data_status.append(True)
            # dataset['data_status'][i] = True
        else:
            new_data_status.append(dataset['data_status'][i])

    print new_data_status.__len__()
    dataset['data_status'] = pd.Series(new_data_status)
    print sum(dataset['data_status'].isnull())

    dataset.to_csv(dataset_path,index=False)

"""
调整样本分布
"""
def sample_weight_adjustment_011():
    # weight011
    dataset_path = r'E:\work_file\mmt_application_card\raw_data\mmt_application_model_feature_f.csv'
    dataset_train_path1 = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight011\mmt_acard_feature_train1.csv'
    dataset_train_path2 = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight011\mmt_acard_feature_train2.csv'
    dataset_train_path3 = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight011\mmt_acard_feature_train3.csv'

    dataset = pd.read_csv(dataset_path)
    print 'TRAIN DATASET:',dataset[dataset.dataset_split=='train'].shape
    print dataset[dataset.dataset_split=='train'].groupby('target1').count().iloc[:,1]
    print dataset[dataset.dataset_split=='train'].groupby('target2').count().iloc[:,1]
    print dataset[dataset.dataset_split=='train'].groupby('target3').count().iloc[:,1]

    print '####target1######'
    dataset_tmp = dataset[dataset.dataset_split=='train']
    dataset_tmp['target'] = dataset_tmp['target1']
    dataset_tmp = dataset_tmp[dataset_tmp.application_comp_date>='2017-08-01']
    print 'synthetic sample shape:',dataset_tmp.shape
    print 'EXPT TO:',dataset_train_path1
    dataset_tmp.to_csv(dataset_train_path1,index=False)

    print '####target2######'
    dataset_tmp = dataset[dataset.dataset_split=='train']
    dataset_tmp['target'] = dataset_tmp['target2']
    dataset_tmp = dataset_tmp[dataset_tmp.application_comp_date>='2017-08-01']
    print 'synthetic sample shape:',dataset_tmp.shape
    print 'EXPT TO:',dataset_train_path2
    dataset_tmp.to_csv(dataset_train_path2,index=False)

    print '####target3######'
    dataset_tmp = dataset[dataset.dataset_split=='train']
    dataset_tmp['target'] = dataset_tmp['target3']
    dataset_tmp = dataset_tmp[dataset_tmp.application_comp_date>='2017-08-01']
    print 'synthetic sample shape:',dataset_tmp.shape
    print 'EXPT TO:',dataset_train_path3
    dataset_tmp.to_csv(dataset_train_path3,index=False)


def sample_weight_adjustment_012():
    # weight012
    dataset_path = r'E:\work_file\mmt_application_card\raw_data\mmt_application_model_feature_f.csv'
    dataset_train_path1 = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight012\mmt_acard_feature_train1.csv'
    dataset_train_path2 = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight012\mmt_acard_feature_train2.csv'
    dataset_train_path3 = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight012\mmt_acard_feature_train3.csv'

    dataset = pd.read_csv(dataset_path)
    print 'TRAIN DATASET:',dataset[dataset.dataset_split=='train'].shape
    print dataset[dataset.dataset_split=='train'].groupby('target1').count().iloc[:,1]
    print dataset[dataset.dataset_split=='train'].groupby('target2').count().iloc[:,1]
    print dataset[dataset.dataset_split=='train'].groupby('target3').count().iloc[:,1]

    print '####target1######'
    dataset_tmp = dataset[dataset.dataset_split=='train']
    dataset_tmp['target'] = dataset_tmp['target1']
    dataset_tmp = dataset_tmp[dataset_tmp.application_comp_date>='2017-08-01']
    df_list = []
    df_list.append(dataset_tmp)
    df_list.append(dataset_tmp[dataset_tmp.application_comp_date>='2017-11-01'])
    synthetic_df = pd.concat(df_list,ignore_index=True)
    print 'synthetic sample shape:',synthetic_df.shape
    print 'EXPT TO:',dataset_train_path1
    synthetic_df.to_csv(dataset_train_path1,index=False)

    print '####target2######'
    dataset_tmp = dataset[dataset.dataset_split=='train']
    dataset_tmp['target'] = dataset_tmp['target2']
    dataset_tmp = dataset_tmp[dataset_tmp.application_comp_date>='2017-08-01']
    df_list = []
    df_list.append(dataset_tmp)
    df_list.append(dataset_tmp[dataset_tmp.application_comp_date>='2017-11-01'])
    synthetic_df = pd.concat(df_list,ignore_index=True)
    print 'synthetic sample shape:',synthetic_df.shape
    print 'EXPT TO:',dataset_train_path2
    synthetic_df.to_csv(dataset_train_path2,index=False)

    print '####target3######'
    dataset_tmp = dataset[dataset.dataset_split=='train']
    dataset_tmp['target'] = dataset_tmp['target3']
    dataset_tmp = dataset_tmp[dataset_tmp.application_comp_date>='2017-08-01']
    df_list = []
    df_list.append(dataset_tmp)
    df_list.append(dataset_tmp[dataset_tmp.application_comp_date>='2017-11-01'])
    synthetic_df = pd.concat(df_list,ignore_index=True)
    print 'synthetic sample shape:',synthetic_df.shape
    print 'EXPT TO:',dataset_train_path3
    synthetic_df.to_csv(dataset_train_path3,index=False)


def sample_weight_adjustment_123():
    # weight123
    dataset_path = r'E:\work_file\mmt_application_card\raw_data\mmt_application_model_feature_f.csv'
    dataset_train_path1 = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight123\mmt_acard_feature_train1.csv'
    dataset_train_path2 = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight123\mmt_acard_feature_train2.csv'
    dataset_train_path3 = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight123\mmt_acard_feature_train3.csv'

    dataset = pd.read_csv(dataset_path)
    print 'TRAIN DATASET:',dataset[dataset.dataset_split=='train'].shape
    print dataset[dataset.dataset_split=='train'].groupby('target1').count().iloc[:,1]
    print dataset[dataset.dataset_split=='train'].groupby('target2').count().iloc[:,1]
    print dataset[dataset.dataset_split=='train'].groupby('target3').count().iloc[:,1]

    print '####target1######'
    dataset_tmp = dataset[dataset.dataset_split=='train']
    dataset_tmp['target'] = dataset_tmp['target1']
    df_list = []
    df_list.append(dataset_tmp)
    df_list.append(dataset_tmp[dataset_tmp.application_comp_date>='2017-08-01'])
    df_list.append(dataset_tmp[dataset_tmp.application_comp_date>='2017-11-01'])
    synthetic_df = pd.concat(df_list,ignore_index=True)
    print 'synthetic sample shape:',synthetic_df.shape
    print 'EXPT TO:',dataset_train_path1
    synthetic_df.to_csv(dataset_train_path1,index=False)

    print '####target2######'
    dataset_tmp = dataset[dataset.dataset_split=='train']
    dataset_tmp['target'] = dataset_tmp['target2']
    df_list = []
    df_list.append(dataset_tmp)
    df_list.append(dataset_tmp[dataset_tmp.application_comp_date>='2017-08-01'])
    df_list.append(dataset_tmp[dataset_tmp.application_comp_date>='2017-11-01'])
    synthetic_df = pd.concat(df_list,ignore_index=True)
    print 'synthetic sample shape:',synthetic_df.shape
    print 'EXPT TO:',dataset_train_path2
    synthetic_df.to_csv(dataset_train_path2,index=False)

    print '####target3######'
    dataset_tmp = dataset[dataset.dataset_split=='train']
    dataset_tmp['target'] = dataset_tmp['target3']
    df_list = []
    df_list.append(dataset_tmp)
    df_list.append(dataset_tmp[dataset_tmp.application_comp_date>='2017-08-01'])
    df_list.append(dataset_tmp[dataset_tmp.application_comp_date>='2017-11-01'])
    synthetic_df = pd.concat(df_list,ignore_index=True)
    print 'synthetic sample shape:',synthetic_df.shape
    print 'EXPT TO:',dataset_train_path3
    synthetic_df.to_csv(dataset_train_path3,index=False)

if __name__ == '__main__':
    sample_weight_adjustment_011()
    sample_weight_adjustment_012()
    sample_weight_adjustment_123()









