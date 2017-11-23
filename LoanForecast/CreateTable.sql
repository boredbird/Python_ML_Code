CREATE TABLE `t_user` (
  `uid` VARCHAR(10) DEFAULT NULL,
  `age` INT(4) DEFAULT NULL,
  `sex` INT(4) DEFAULT NULL,
  `active_date` date DEFAULT NULL,
  `limit` double(12,10) DEFAULT NULL
) ENGINE=INNODB DEFAULT CHARSET=utf8 ;


CREATE TABLE `t_order` (
  `uid` VARCHAR(10) DEFAULT NULL,
  `buy_time` date DEFAULT NULL,
  `price` double(12,10) DEFAULT NULL,
  `qty` INT(4) DEFAULT NULL,
  `cate_id` INT(4) DEFAULT NULL,
  `discount` double(12,10) DEFAULT NULL
) ENGINE=INNODB DEFAULT CHARSET=utf8 ;

CREATE TABLE `t_click` (
  `uid` VARCHAR(10) DEFAULT NULL,
  `click_time` datetime DEFAULT NULL,
  `pid` INT(8) DEFAULT NULL,
  `param` INT(8) DEFAULT NULL
) ENGINE=INNODB DEFAULT CHARSET=utf8 ;

CREATE TABLE `t_loan` (
  `uid` VARCHAR(10) DEFAULT NULL,
  `loan_time` datetime DEFAULT NULL,
  `loan_amount` double(12,10) DEFAULT NULL,
  `plannum` INT(8) DEFAULT NULL
) ENGINE=INNODB DEFAULT CHARSET=utf8 ;

CREATE TABLE `t_loan_sum` (
  `uid` VARCHAR(10) DEFAULT NULL,
  `month` VARCHAR(10) DEFAULT NULL,
  `loan_sum` double(12,10) DEFAULT NULL
) ENGINE=INNODB DEFAULT CHARSET=utf8 ;

select count(distinct uid),count(distinct pid),count(distinct param),count(*) from `t_click`
78439	10	48	10933016

select COUNT(DISTINCT uid),COUNT(DISTINCT plannum),COUNT(*) from `t_loan`
36697	4	202902

select COUNT(DISTINCT uid),COUNT(DISTINCT month),COUNT(*) from `t_loan_sum`
19520	1	19520

select COUNT(DISTINCT uid),COUNT(DISTINCT qty),COUNT(DISTINCT cate_id),COUNT(*) from t_order
89244	2425	44	5400778

select COUNT(DISTINCT uid),min(age),max(age),COUNT(DISTINCT sex),COUNT(*) from t_user
90993	20	50	2	90993

ALTER TABLE `dmc`.`t_user`   
  ADD  INDEX `idx_uid` (`uid`);
ALTER TABLE `dmc`.`t_click`   
  ADD  INDEX `idx_uid` (`uid`);
ALTER TABLE `dmc`.`t_loan`   
  ADD  INDEX `idx_uid` (`uid`);
ALTER TABLE `dmc`.`t_loan_sum`   
  ADD  INDEX `idx_uid` (`uid`);
ALTER TABLE `dmc`.`t_order`   
  ADD  INDEX `idx_uid` (`uid`);
  

 SELECT MIN(loan_amount),MAX(loan_amount),AVG(loan_amount) FROM `t_loan`
 MIN(loan_amount) MAX(loan_amount)  AVG(loan_amount)
2.8675356043  7.5840624551  4.91341061925787

 SELECT MIN(loan_sum),MAX(loan_sum),AVG(loan_sum) FROM  t_loan_sum
 MIN(loan_sum)  MAX(loan_sum) AVG(loan_sum)
3.8625945450  8.4330190930  5.56896173850699


count(*)

daydiff activate_date loan_date

uid
sex
age

cate_id
plannum
pid
param

1、利用t_order和t_click的信息，预测明细表汇总数据 与汇总表数据的关系
2、预测用户是否借款
3、预测用户借款金额

超参数：分类模型的概率值，召回率，准确率，只有预测为借款的用户才计算回归值，预测为不借款的用户给出的预测值为0

每月应还款

浏览过的商品的均价price

暴力加工特征是否行得通？

维度：group by 
数值：sum avg min max count 

衍生占比类特征

样本时间窗口敲定


SELECT cnt,COUNT(*)
FROM
(SELECT uid,COUNT(*) cnt
FROM `t_loan`  
WHERE SUBSTR(loan_time,1,7)='2016-11' 
GROUP BY uid
) tt
GROUP BY cnt
 