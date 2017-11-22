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
  








