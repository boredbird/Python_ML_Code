# -*- coding:utf-8 -*-
__author__ = 'chunhui.zhang'
import pandas as pd
import load_data as ld

# raw data file path
rawdata_path = 'E:/ScoreCard/rawdata/'
# raw data file name
raw_file_name = ['pos_model_var_tbl_train', 'pos_model_var_tbl_validation']
train_file_name = 'pos_model_var_tbl_train'

# split data path
split_data_path = ''

# model path
model_path = 'E:/Code/ScoreCard/'

# feature path
feature_path = ''

# gen data path
gen_data_path = 'E:/ScoreCard/gendata/'
# gen data file name
gendata_file_name = 'dataset_train_woe'

# output path
output_path = 'E:/Code/ScoreCard/output/'


# specify variable data type
# variable_type = [['cus_id','int64']
# ,['observe_date','object']
# ,['cus_sex','object']
# ,['cus_age','int64']
# ,['cus_cert_initial4','object']
# ,['has_pos_loan','int64']
# ,['hist_other_pay_cnt','int64']
# ,['has_prepay_pos','int64']
# ,['application_cnt_pos','int64']
# ,['approve_cnt_pos','int64']
# ,['max_credit_pos','int64']
# ,['max_instalment_pos','int64']
# ,['max_hist_cpd_pos','int64']
# ,['active_loan_cnt_pos','int64']
# ,['pos_app_date','object']
# ,['pos_contract_no','object']
# ,['pos_credit','int64']
# ,['pos_period_cnt','int64']
# ,['pos_other_contact_type','object']
# ,['pos_relative_type','object']
# ,['pos_due_periods_ratio','float64']
# ,['pos_finished_periods_cnt','int64']
# ,['pos_dd_fail_cnt','int64']
# ,['pos_dd_fail_ratio','float64']
# ,['pos_on_time_pay_cnt','int64']
# ,['pos_in_time_pay_cnt','int64']
# ,['pos_total_delay_day_cnt','int64']
# ,['pos_sales_commission','float64']
# ,['pos_sa_city','object']
# ,['pos_cur_banlance','float64']
# ,['sa_p_r1pd30_pos','float64']
# ,['sa_p_r2pd30_pos','float64']
# ,['sa_p_r3pd30_pos','float64']
# ,['sa_p_r4pd30_pos','float64']
# ,['sa_p_r5pd30_pos','float64']
# ,['sa_p_rcpd90_7_pos','float64']
# ,['sa_pos_appr_rate_pos','float64']
# ,['sa_avg_apply_amount_pos','float64']
# ,['sa_avg_payment_num_pos','int64']
# ,['sa_over4k_amount_rate_pos','float64']
# ,['sa_avg_tcprice_pos','int64']
# ,['sa_inaugurate_date_pos','object']
# ,['sa_cert_initial4_pos','object']
# ,['sa_cert_initial6_pos','object']
# ,['sa_level_pos','object']
# ,['sa_status_pos','object']
# ,['shop_type_pos','object']
# ,['shop_first_loan_date_pos','object']
# ,['shop_status_pos','object']
# ,['dd1_pos','int64']
# ,['dd2_pos','int64']
# ,['dd3_pos','int64']
# ,['dd4_pos','int64']
# ,['dd5_pos','int64']
# ,['dd6_pos','int64']
# ,['dd7_pos','int64']
# ,['dd8_pos','int64']
# ,['target','int64']]

#convert pandas list to DataFrame
#and set v_name be index
# variable_type = pd.DataFrame(variable_type,columns=['v_name','v_type'])
# variable_type=variable_type.set_index(['v_name'])

#load config file
config =ld.load_from_csv(model_path,['config',])
config = config[0]


