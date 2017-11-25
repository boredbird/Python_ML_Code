--------------------------------------------------用户特征---------------------------------------------
--基本属性--用户信息表
SELECT t.uid
,t.`age` --年龄段
,t.`sex` --	性别
,t.`limit` --初始额度
,DATEDIFF('2017-01-01', active_date) AS daydiff_active --用户激活天数
FROM `t_user` t;

--用户下单特征--订单信息表--GROUP BY uid
CREATE TABLE user_order_tmp01
AS
SELECT 
uid
,count(*) as uid_order_cnt
,SUM(price)  AS uid_price_sum
,SUM(discount) AS uid_discount_sum
,sum(qty) as uid_qty_sum
,avg(qty) as uid_qty_avg
,min(qty) as uid_qty_min
,max(qty) as uid_qty_max
,AVG(price)	AS uid_price_avg
,AVG(discount) AS uid_discount_avg
,COUNT(DISTINCT qty) AS uid_qty_cnt
,COUNT(DISTINCT cate_id) AS uid_cate_cnt
,MIN(buy_time) AS uid_buy_time_min
,MAX(buy_time) AS uid_buy_time_max
,DATEDIFF(MAX(buy_time), MIN(buy_time)) AS uid_buy_time_daydiff
FROM `t_order`
GROUP BY uid;

CREATE TABLE user_order_cate_tmp01
AS
SELECT 
uid
,cate_id
,COUNT(*) AS uid_cate_order_cnt
,SUM(price)  AS uid_cate_price_sum
,SUM(discount) AS uid_cate_discount_sum
,SUM(qty) AS uid_cate_qty_sum
,AVG(qty) AS uid_cate_qty_avg
,AVG(price)	AS uid_cate_price_avg
,AVG(discount) AS uid_cate_discount_avg
,min(qty) as uid_cate_qty_min
,MIN(buy_time) AS uid_cate_buy_time_min
,max(qty) as uid_cate_qty_max
,MAX(buy_time) AS uid_cate_buy_time_max
,COUNT(DISTINCT qty) AS uid_cate_qty_cnt
,COUNT(DISTINCT cate_id) AS uid_cate_cate_cnt

,DATEDIFF(MAX(buy_time), MIN(buy_time)) AS uid_cate_buy_time_daydiff
FROM `t_order`
GROUP BY uid,cate_id;

--用户借款特征--借款信息表--GROUP BY uid
CREATE TABLE user_loan_tmp01
AS
SELECT 
uid
,MIN(loan_time) AS uid_loan_time_min
,MIN(loan_amount) AS uid_loan_amount_min
,MIN(plannum) AS uid_plannum_min
,MAX(loan_time) AS uid_loan_time_max
,MAX(loan_amount) AS uid_loan_amount_max
,MAX(plannum) AS uid_plannum_max
,SUM(loan_amount) AS uid_loan_amount_sum
,SUM(plannum) AS uid_plannum_sum
,AVG(loan_amount) AS uid_loan_amount_avg
,AVG(plannum) AS uid_plannum_avg
,DATEDIFF(MAX(loan_time), MIN(loan_time)) AS uid_loan_time_daydiff
FROM `t_loan`
GROUP BY uid;

CREATE TABLE user_loan_plannum_tmp01
AS
SELECT 
uid
,plannum
,MIN(loan_time) AS uid_plannum_loan_time_min
,MIN(loan_amount) AS uid_plannum_loan_amount_min
,MAX(loan_amount) AS uid_plannum_loan_amount_max
,MAX(loan_time) AS uid_plannum_loan_time_max
,DATEDIFF(MAX(loan_time), MIN(loan_time)) AS uid_plannum_loan_time_daydiff
,SUM(loan_amount) AS uid_plannum_loan_amount_sum
,AVG(loan_amount) AS uid_plannum_loan_amount_avg
FROM `t_loan`
GROUP BY uid,plannum;


--用户行为特征--点击信息表--GROUP BY uid
CREATE TABLE user_click_tmp01
AS
SELECT uid
,COUNT(*) AS user_click_cnt
,MIN(click_time) AS uid_click_time_min
,MAX(click_time) AS uid_click_time_max
,DATEDIFF(MAX(click_time), MIN(click_time)) AS uid_click_time_daydiff
,COUNT(DISTINCT pid) AS user_click_pid_cnt
,COUNT(DISTINCT param) AS user_click_param_cnt
FROM `t_click`
GROUP BY uid

CREATE TABLE user_click_pid_tmp01
AS
SELECT uid
,pid
,COUNT(*) AS user_click_pid_cnt
,MIN(click_time) AS uid_pid_click_time_min
,MAX(click_time) AS uid_pid_click_time_max
,DATEDIFF(MAX(click_time), MIN(click_time)) AS uid_pid_click_time_daydiff
,COUNT(DISTINCT param) AS user_click_pid_param_cnt
FROM `t_click`
GROUP BY uid,pid;

CREATE TABLE user_click_param_tmp01
AS
SELECT uid
,param
,COUNT(*) AS user_click_param_cnt
,MIN(click_time) AS uid_param_click_time_min
,MAX(click_time) AS uid_param_click_time_max
,DATEDIFF(MAX(click_time), MIN(click_time)) AS uid_param_click_time_daydiff
,COUNT(DISTINCT pid) AS user_click_param_pid_cnt
FROM `t_click`
GROUP BY uid,param;

CREATE TABLE user_click_param_pid_tmp01
AS
SELECT uid
,pid
,param
,COUNT(*) AS user_click_param_pid_cnt
,MIN(click_time) AS uid_param_pid_click_time_min
,MAX(click_time) AS uid_param_pid_click_time_max
,DATEDIFF(MAX(click_time), MIN(click_time)) AS uid_param_pid_click_time_daydiff
FROM `t_click`
GROUP BY uid,pid,param;


--------------------------------------------------年龄段特征---------------------------------------------
--用户信息表 -- group by age
CREATE TABLE age_tmp01
AS
SELECT 
age
,COUNT(*) AS age_cnt
,COUNT(DISTINCT sex) AS age_sex_cnt
,MIN(active_date) AS age_active_date_min
,MAX(active_date) AS age_active_date_max
,DATEDIFF(MAX(active_date), MIN(active_date)) AS age_active_date_daydiff
,SUM(LIMIT) AS age_limit_sum
,AVG(LIMIT) AS age_limit_avg
,MIN(LIMIT) AS age_limit_min
,MAX(LIMIT) AS age_limit_max
FROM `t_user`
GROUP BY age;

CREATE TABLE age_sex_tmp01
AS
SELECT 
age
,sex
,COUNT(*) AS age_sex_cnt
,MIN(active_date) AS age_sex_active_date_min
,MAX(active_date) AS age_sex_active_date_max
,DATEDIFF(MAX(active_date), MIN(active_date)) AS age_sex_active_date_daydiff
,SUM(LIMIT) AS age_sex_limit_sum
,AVG(LIMIT) AS age_sex_limit_avg
,MIN(LIMIT) AS age_sex_limit_min
,MAX(LIMIT) AS age_sex_limit_max
FROM `t_user`
GROUP BY age,sex;


CREATE TABLE age_order_tmp01
AS
SELECT 
t2.age
,COUNT(*) AS age_order_cnt
,SUM(price)  AS age_price_sum
,SUM(discount) AS age_discount_sum
,SUM(qty) AS age_qty_sum
,AVG(qty) AS age_qty_avg
,MIN(qty) AS age_qty_min
,MAX(qty) AS age_qty_max
,AVG(price)	AS age_price_avg
,AVG(discount) AS age_discount_avg
,COUNT(DISTINCT qty) AS age_qty_cnt
,COUNT(DISTINCT cate_id) AS age_cate_cnt
,MIN(buy_time) AS age_buy_time_min
,MAX(buy_time) AS age_buy_time_max
,DATEDIFF(MAX(buy_time), MIN(buy_time)) AS age_buy_time_daydiff
FROM `t_order` t1
LEFT JOIN t_user t2 ON t1.`uid`=t2.`uid`
GROUP BY t2.age;

CREATE TABLE age_order_cate_tmp01
AS
SELECT 
t2.age
,cate_id
,COUNT(*) AS uid_cate_order_cnt
,SUM(price)  AS uid_cate_price_sum
,SUM(discount) AS uid_cate_discount_sum
,SUM(qty) AS uid_cate_qty_sum
,AVG(qty) AS uid_cate_qty_avg
,min(qty) as uid_cate_qty_min
,max(qty) as uid_cate_qty_max
,AVG(price)	AS uid_cate_price_avg
,AVG(discount) AS uid_cate_discount_avg
,COUNT(DISTINCT qty) AS uid_cate_qty_cnt
,COUNT(DISTINCT cate_id) AS uid_cate_cate_cnt
,MIN(buy_time) AS uid_cate_buy_time_min
,MAX(buy_time) AS uid_cate_buy_time_max
,DATEDIFF(MAX(buy_time), MIN(buy_time)) AS uid_cate_buy_time_daydiff
FROM `t_order` t1
left join t_user t2 on t1.uid=t2.uid
GROUP BY t2.age,cate_id;

CREATE TABLE age_loan_tmp01
AS
SELECT 
t2.age
,MIN(loan_time) AS age_loan_time_min
,MAX(loan_time) AS age_loan_time_max
,DATEDIFF(MAX(loan_time), MIN(loan_time)) AS age_loan_time_daydiff
,SUM(loan_amount) AS age_loan_amount_sum
,AVG(loan_amount) AS age_loan_amount_avg
,MIN(loan_amount) AS age_loan_amount_min
,MAX(loan_amount) AS age_loan_amount_max
,MIN(plannum) AS age_plannum_min
,MAX(plannum) AS age_plannum_max
,AVG(plannum) AS age_plannum_avg
,SUM(plannum) AS age_plannum_sum
FROM `t_loan` t1
left join t_user t2 on t1.uid=t2.uid
GROUP BY t2.age;

CREATE TABLE age_loan_plannum_tmp01
AS
SELECT 
t2.age
,plannum
,MIN(loan_time) AS age_plannum_loan_time_min
,MAX(loan_time) AS age_plannum_loan_time_max
,DATEDIFF(MAX(loan_time), MIN(loan_time)) AS age_plannum_loan_time_daydiff
,SUM(loan_amount) AS age_plannum_loan_amount_sum
,AVG(loan_amount) AS age_plannum_loan_amount_avg
,MIN(loan_amount) AS age_plannum_loan_amount_min
,MAX(loan_amount) AS age_plannum_loan_amount_max
FROM `t_loan` t1
left join t_user t2 on t1.uid=t2.uid
GROUP BY t2.age,plannum;

CREATE TABLE age_click_tmp01
AS
SELECT t2.age
,COUNT(*) AS age_click_cnt
,MIN(click_time) AS uid_click_time_min
,MAX(click_time) AS uid_click_time_max
,DATEDIFF(MAX(click_time), MIN(click_time)) AS uid_click_time_daydiff
,COUNT(DISTINCT pid) AS age_click_pid_cnt
,COUNT(DISTINCT param) AS age_click_param_cnt
FROM `t_click` t1
LEFT JOIN t_user t2 ON t1.`uid`=t2.`uid`
GROUP BY t2.age;

CREATE TABLE age_click_pid_tmp01
AS
SELECT t2.age
,pid
,COUNT(*) AS age_click_pid_cnt
,MIN(click_time) AS uid_pid_click_time_min
,MAX(click_time) AS uid_pid_click_time_max
,DATEDIFF(MAX(click_time), MIN(click_time)) AS uid_pid_click_time_daydiff
,COUNT(DISTINCT param) AS age_click_pid_param_cnt
FROM `t_click` t1
LEFT JOIN t_user t2 ON t1.`uid`=t2.`uid`
GROUP BY t2.age,pid;

CREATE TABLE age_click_param_tmp01
AS
SELECT t2.age
,param
,COUNT(*) AS age_click_param_cnt
,MIN(click_time) AS uid_param_click_time_min
,MAX(click_time) AS uid_param_click_time_max
,DATEDIFF(MAX(click_time), MIN(click_time)) AS uid_param_click_time_daydiff
,COUNT(DISTINCT pid) AS age_click_param_pid_cnt
FROM `t_click` t1
LEFT JOIN t_user t2 ON t1.`uid`=t2.`uid`
GROUP BY t2.age,param;

CREATE TABLE age_click_param_pid_tmp01
AS
SELECT t2.age
,pid
,param
,COUNT(*) AS age_click_param_pid_cnt
,MIN(click_time) AS uid_param_pid_click_time_min
,MAX(click_time) AS uid_param_pid_click_time_max
,DATEDIFF(MAX(click_time), MIN(click_time)) AS uid_param_pid_click_time_daydiff
FROM `t_click` t1
LEFT JOIN t_user t2 ON t1.`uid`=t2.`uid`
GROUP BY t2.age,pid,param;


--------------------------------------------------性别特征---------------------------------------------
--用户信息表 -- group by age
CREATE TABLE sex_tmp01
AS
SELECT 
sex
,COUNT(*) AS sex_cnt
,MIN(active_date) AS sex_active_date_min
,MIN(LIMIT) AS sex_limit_min
,MAX(active_date) AS sex_active_date_max
,MAX(LIMIT) AS sex_limit_max
,AVG(LIMIT) AS sex_limit_avg
,DATEDIFF(MAX(active_date), MIN(active_date)) AS sex_active_date_daydiff
,SUM(LIMIT) AS sex_limit_sum
FROM `t_user`
GROUP BY sex;

CREATE TABLE sex_order_tmp01
AS
SELECT 
t2.sex
,COUNT(*) AS sex_order_cnt
,SUM(price)  AS sex_price_sum
,SUM(discount) AS sex_discount_sum
,SUM(qty) AS sex_qty_sum
,MIN(qty) AS sex_qty_min
,MIN(buy_time) AS sex_buy_time_min
,MAX(qty) AS sex_qty_max
,MAX(buy_time) AS sex_buy_time_max
,AVG(qty) AS sex_qty_avg
,AVG(price)	AS sex_price_avg
,AVG(discount) AS sex_discount_avg
,COUNT(DISTINCT qty) AS sex_qty_cnt
,COUNT(DISTINCT cate_id) AS sex_cate_cnt
,DATEDIFF(MAX(buy_time), MIN(buy_time)) AS sex_buy_time_daydiff
FROM `t_order` t1
LEFT JOIN t_user t2 ON t1.`uid`=t2.`uid`
GROUP BY t2.sex;

CREATE TABLE sex_order_cate_tmp01
AS
SELECT 
t2.sex
,cate_id
,COUNT(*) AS uid_cate_order_cnt
,SUM(price)  AS uid_cate_price_sum
,SUM(discount) AS uid_cate_discount_sum
,SUM(qty) AS uid_cate_qty_sum
,min(qty) as uid_cate_qty_min
,MIN(buy_time) AS uid_cate_buy_time_min
,max(qty) as uid_cate_qty_max
,MAX(buy_time) AS uid_cate_buy_time_max
,AVG(qty) AS uid_cate_qty_avg
,AVG(price)	AS uid_cate_price_avg
,AVG(discount) AS uid_cate_discount_avg
,COUNT(DISTINCT qty) AS uid_cate_qty_cnt
,COUNT(DISTINCT cate_id) AS uid_cate_cate_cnt
,DATEDIFF(MAX(buy_time), MIN(buy_time)) AS uid_cate_buy_time_daydiff
FROM `t_order` t1
left join t_user t2 on t1.uid=t2.uid
GROUP BY t2.sex,cate_id;

CREATE TABLE sex_loan_tmp01
AS
SELECT 
t2.sex
,MIN(loan_time) AS sex_loan_time_min
,MAX(loan_time) AS sex_loan_time_max
,DATEDIFF(MAX(loan_time), MIN(loan_time)) AS sex_loan_time_daydiff
,SUM(loan_amount) AS sex_loan_amount_sum
,AVG(loan_amount) AS sex_loan_amount_avg
,MIN(loan_amount) AS sex_loan_amount_min
,MAX(loan_amount) AS sex_loan_amount_max
,MIN(plannum) AS sex_plannum_min
,MAX(plannum) AS sex_plannum_max
,AVG(plannum) AS sex_plannum_avg
,SUM(plannum) AS sex_plannum_sum
FROM `t_loan` t1
left join t_user t2 on t1.uid=t2.uid
GROUP BY t2.sex;

CREATE TABLE sex_loan_plannum_tmp01
AS
SELECT 
t2.sex
,plannum
,MIN(loan_time) AS sex_plannum_loan_time_min
,MAX(loan_time) AS sex_plannum_loan_time_max
,DATEDIFF(MAX(loan_time), MIN(loan_time)) AS sex_plannum_loan_time_daydiff
,SUM(loan_amount) AS sex_plannum_loan_amount_sum
,AVG(loan_amount) AS sex_plannum_loan_amount_avg
,MIN(loan_amount) AS sex_plannum_loan_amount_min
,MAX(loan_amount) AS sex_plannum_loan_amount_max
FROM `t_loan` t1
left join t_user t2 on t1.uid=t2.uid
GROUP BY t2.sex,plannum;

CREATE TABLE sex_click_tmp01
AS
SELECT t2.sex
,COUNT(*) AS sex_click_cnt
,MIN(click_time) AS uid_click_time_min
,MAX(click_time) AS uid_click_time_max
,DATEDIFF(MAX(click_time), MIN(click_time)) AS uid_click_time_daydiff
,COUNT(DISTINCT pid) AS sex_click_pid_cnt
,COUNT(DISTINCT param) AS sex_click_param_cnt
FROM `t_click` t1
LEFT JOIN t_user t2 ON t1.`uid`=t2.`uid`
GROUP BY t2.sex;

CREATE TABLE sex_click_pid_tmp01
AS
SELECT t2.sex
,pid
,COUNT(*) AS sex_click_pid_cnt
,MIN(click_time) AS uid_pid_click_time_min
,MAX(click_time) AS uid_pid_click_time_max
,DATEDIFF(MAX(click_time), MIN(click_time)) AS uid_pid_click_time_daydiff
,COUNT(DISTINCT param) AS sex_click_pid_param_cnt
FROM `t_click` t1
LEFT JOIN t_user t2 ON t1.`uid`=t2.`uid`
GROUP BY t2.sex,pid;

CREATE TABLE sex_click_param_tmp01
AS
SELECT t2.sex
,param
,COUNT(*) AS sex_click_param_cnt
,MIN(click_time) AS uid_param_click_time_min
,MAX(click_time) AS uid_param_click_time_max
,DATEDIFF(MAX(click_time), MIN(click_time)) AS uid_param_click_time_daydiff
,COUNT(DISTINCT pid) AS sex_click_param_pid_cnt
FROM `t_click` t1
LEFT JOIN t_user t2 ON t1.`uid`=t2.`uid`
GROUP BY t2.sex,param;

CREATE TABLE sex_click_param_pid_tmp01
AS
SELECT t2.sex
,pid
,param
,COUNT(*) AS sex_click_param_pid_cnt
,MIN(click_time) AS uid_param_pid_click_time_min
,MAX(click_time) AS uid_param_pid_click_time_max
,DATEDIFF(MAX(click_time), MIN(click_time)) AS uid_param_pid_click_time_daydiff
FROM `t_click` t1
LEFT JOIN t_user t2 ON t1.`uid`=t2.`uid`
GROUP BY t2.sex,pid,param;


--------------------------------------------------cate特征---------------------------------------------
CREATE TABLE cate_id_order_tmp01
AS
SELECT 
cate_id
,count(*) as cate_id_order_cnt
,SUM(price)  AS cate_id_price_sum
,AVG(price)	AS cate_id_price_avg
,SUM(discount) AS cate_id_discount_sum
,AVG(discount) AS cate_id_discount_avg
,sum(qty) as cate_id_qty_sum
,avg(qty) as cate_id_qty_avg
,min(qty) as cate_id_qty_min
,max(qty) as cate_id_qty_max
,COUNT(DISTINCT qty) AS cate_id_qty_cnt
,COUNT(DISTINCT cate_id) AS cate_id_cate_cnt
,MIN(buy_time) AS cate_id_buy_time_min
,MAX(buy_time) AS cate_id_buy_time_max
,DATEDIFF(MAX(buy_time), MIN(buy_time)) AS cate_id_buy_time_daydiff
FROM `t_order`
GROUP BY cate_id;

