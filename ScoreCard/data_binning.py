# -*- coding:utf-8 -*-
__author__ = 'maomaochong'
import sys
sys.path.append(r'E:\Code\Python_ML_Code\ScoreCard')
import load_data as ld
import pandas as pd

rawdata_path = 'E:/ScoreCard/rowdata/'
dataset =ld.load_from_csv(rawdata_path,['pos_model_var_tbl_train',])
dataset_train = dataset[0]

dataset_train['cus_age'] = dataset_train['cus_age'].astype(int)

a = [['cus_id','int64']
,['observe_date','object']
,['cus_sex','int64']
,['cus_age','int64']
,['cus_cert_initial4','int64']
,['has_pos_loan','int64']
,['hist_other_pay_cnt','int64']
,['has_prepay_pos','int64']
,['application_cnt_pos','int64']
,['approve_cnt_pos','int64']
,['max_credit_pos','object']
,['max_instalment_pos','object']
,['max_hist_cpd_pos','int64']
,['active_loan_cnt_pos','int64']
,['pos_app_date','object']
,['pos_contract_no','int64']
,['pos_credit','object']
,['pos_period_cnt','int64']
,['pos_other_contact_type','object']
,['pos_relative_type','object']
,['pos_due_periods_ratio','float64']
,['pos_finished_periods_cnt','int64']
,['pos_dd_fail_cnt','int64']
,['pos_dd_fail_ratio','float64']
,['pos_on_time_pay_cnt','int64']
,['pos_in_time_pay_cnt','int64']
,['pos_total_delay_day_cnt','int64']
,['pos_sales_commission','float64']
,['pos_sa_city','object']
,['pos_cur_banlance','object']
,['sa_p_r1pd30_pos','float64']
,['sa_p_r2pd30_pos','float64']
,['sa_p_r3pd30_pos','float64']
,['sa_p_r4pd30_pos','float64']
,['sa_p_r5pd30_pos','float64']
,['sa_p_rcpd90_7_pos','float64']
,['sa_pos_appr_rate_pos','float64']
,['sa_avg_apply_amount_pos','object']
,['sa_avg_payment_num_pos','int64']
,['sa_over4k_amount_rate_pos','float64']
,['sa_avg_tcprice_pos','int64']
,['sa_inaugurate_date_pos','object']
,['sa_cert_initial4_pos','int64']
,['sa_cert_initial6_pos','int64']
,['sa_level_pos','object']
,['sa_status_pos','object']
,['shop_type_pos','object']
,['shop_first_loan_date_pos','object']
,['shop_status_pos','object']
,['dd1_pos','int64']
,['dd2_pos','int64']
,['dd3_pos','int64']
,['dd4_pos','int64']
,['dd5_pos','int64']
,['dd6_pos','int64']
,['dd7_pos','int64']
,['dd8_pos','int64']
,['target','int64']]

b = pd.DataFrame(a,columns=['v_name','v_type'])\
b=b.set_index(['v_name'])
print b.loc['pos_cur_banlance','v_type'] == 'object'

if len(dataset_train.columns) == b.shape[0]:
    for vname in dataset_train.columns:
        dataset_train[vname] = dataset_train[vname].astype(b.loc[vname,'v_type'])
else:
    print 'error'



