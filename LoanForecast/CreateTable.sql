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

