/*
建表导入数据
*/
DROP TABLE `ccf_first_round_shop_info`;
CREATE TABLE `ccf_first_round_shop_info` (
  `shop_id` VARCHAR(20) DEFAULT NULL COMMENT '店铺ID',
  `category_id` VARCHAR(20) DEFAULT NULL COMMENT '店铺类型ID',
  `longitude` DOUBLE(12,4) DEFAULT NULL COMMENT '店铺位置-经度',
  `latitude` DOUBLE(12,4) DEFAULT NULL COMMENT '店铺位置-纬度',
  `price` INT(8) DEFAULT NULL COMMENT '人均消费指数',
  `mall_id` VARCHAR(20) DEFAULT NULL COMMENT '店铺所在商场ID'
) ENGINE=INNODB DEFAULT CHARSET=utf8 COMMENT='店铺和商场信息表';

DROP TABLE `ccf_first_round_user_shop_behavior`;
CREATE TABLE `ccf_first_round_user_shop_behavior` (
  `user_id` VARCHAR(20) DEFAULT NULL COMMENT '用户ID',
  `shop_id` VARCHAR(20) DEFAULT NULL COMMENT '用户所在店铺ID',
  `time_stamp` DATETIME DEFAULT NULL COMMENT '行为时间戳',
  `longitude` DOUBLE(12,4) DEFAULT NULL COMMENT '行为发生时位置-经度',
  `latitude` DOUBLE(12,4) DEFAULT NULL COMMENT '行为发生时位置-纬度',
  `wifi_infos` VARCHAR(1000) DEFAULT NULL COMMENT '行为发生时Wifi环境，包括bssid（wifi唯一识别码），signal（强度），flag（是否连接）'
) ENGINE=INNODB DEFAULT CHARSET=utf8 COMMENT='用户在店铺内交易表';

DROP TABLE `evaluation_public`;
CREATE TABLE `evaluation_public` (
  `row_id` VARCHAR(20) DEFAULT NULL COMMENT '测试数据ID',
  `user_id` VARCHAR(20) DEFAULT NULL COMMENT '用户ID',
  `mall_id` DOUBLE(12,4) DEFAULT NULL COMMENT '商场ID',
  `time_stamp` DATETIME DEFAULT NULL COMMENT '行为时间戳',
  `longitude` DOUBLE(12,4) DEFAULT NULL COMMENT '行为发生时位置-经度',
  `latitude` DOUBLE(12,4) DEFAULT NULL COMMENT '行为发生时位置-纬度',
  `wifi_infos` VARCHAR(1000) DEFAULT NULL COMMENT '行为发生时Wifi环境'
) ENGINE=INNODB DEFAULT CHARSET=utf8 COMMENT='店铺和商场信息表';


SELECT COUNT(*),COUNT(DISTINCT shop_id),COUNT(DISTINCT category_id),COUNT(DISTINCT mall_id) FROM ccf_first_round_shop_info;
/*
count(*)	count(distinct shop_id)	count(distinct category_id)	COUNT(DISTINCT mall_id)
8477	8477	67	97
*/

SELECT COUNT(*),COUNT(DISTINCT user_id),COUNT(DISTINCT shop_id) FROM ccf_first_round_user_shop_behavior;
/*
COUNT(*)	COUNT(DISTINCT user_id)	COUNT(DISTINCT shop_id)
1138015	714608	8477
*/

SELECT COUNT(*),COUNT(DISTINCT user_id),COUNT(DISTINCT mall_id) FROM evaluation_public;
/*
COUNT(*)	COUNT(DISTINCT user_id)	COUNT(DISTINCT mall_id)
483931	338642	1
*/

SELECT MAX(LENGTH(wifi_infos)) FROM `ccf_first_round_user_shop_behavior`;--419

SELECT * FROM ccf_first_round_user_shop_behavior;







