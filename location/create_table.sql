/*
建表导入数据
*/
DROP TABLE `ccf_first_round_shop_info`;
CREATE TABLE `ccf_first_round_shop_info` (
  `shop_id` VARCHAR(20) DEFAULT NULL COMMENT '店铺ID',
  `category_id` VARCHAR(20) DEFAULT NULL COMMENT '店铺类型ID',
  `longitude` DOUBLE(12,6) DEFAULT NULL COMMENT '店铺位置-经度',
  `latitude` DOUBLE(12,6) DEFAULT NULL COMMENT '店铺位置-纬度',
  `price` INT(8) DEFAULT NULL COMMENT '人均消费指数',
  `mall_id` VARCHAR(20) DEFAULT NULL COMMENT '店铺所在商场ID'
) ENGINE=INNODB DEFAULT CHARSET=utf8 COMMENT='店铺和商场信息表';

DROP TABLE `ccf_first_round_user_shop_behavior`;
CREATE TABLE `ccf_first_round_user_shop_behavior` (
  `user_id` VARCHAR(20) DEFAULT NULL COMMENT '用户ID',
  `shop_id` VARCHAR(20) DEFAULT NULL COMMENT '用户所在店铺ID',
  `time_stamp` DATETIME DEFAULT NULL COMMENT '行为时间戳',
  `longitude` DOUBLE(12,6) DEFAULT NULL COMMENT '行为发生时位置-经度',
  `latitude` DOUBLE(12,6) DEFAULT NULL COMMENT '行为发生时位置-纬度',
  `wifi_infos` VARCHAR(1000) DEFAULT NULL COMMENT '行为发生时Wifi环境，包括bssid（wifi唯一识别码），signal（强度），flag（是否连接）'
) ENGINE=INNODB DEFAULT CHARSET=utf8 COMMENT='用户在店铺内交易表';

DROP TABLE `evaluation_public`;
CREATE TABLE `evaluation_public` (
  `row_id` VARCHAR(20) DEFAULT NULL COMMENT '测试数据ID',
  `user_id` VARCHAR(20) DEFAULT NULL COMMENT '用户ID',
  `mall_id` VARCHAR(20) DEFAULT NULL COMMENT '商场ID',
  `time_stamp` DATETIME DEFAULT NULL COMMENT '行为时间戳',
  `longitude` DOUBLE(12,6) DEFAULT NULL COMMENT '行为发生时位置-经度',
  `latitude` DOUBLE(12,6) DEFAULT NULL COMMENT '行为发生时位置-纬度',
  `wifi_infos` VARCHAR(1000) DEFAULT NULL COMMENT '行为发生时Wifi环境'
) ENGINE=INNODB DEFAULT CHARSET=utf8 COMMENT='店铺和商场信息表';

CREATE TABLE `sequence` (
  `seq` INT(4) DEFAULT NULL COMMENT '序号'
) ENGINE=INNODB DEFAULT CHARSET=utf8 COMMENT='序列号表';

-- SELECT DISTINCT t1.conn_name,t1.id_transformation,t1.id_step,LOWER(t1.sub_id) AS sub_id
-- FROM
-- (SELECT
--     id,
--     SUBSTRING_INDEX(
--         SUBSTRING_INDEX(t.`sql_content`, ',', seq),
--         ',' ,- 1
--     ) sub_id,
--     seq,
--     t.id_transformation,id_step,conn_name
-- FROM
--     sequence
-- CROSS JOIN (SELECT * FROM kettle.etl_step_sql q ) t
-- WHERE
--     seq BETWEEN 1 AND (SELECT 1 + LENGTH(t.`sql_content`) - LENGTH(REPLACE(t.`sql_content`, ',', '')))
-- ORDER BY
--     id,
--     sub_id
--  ) t1     
--  INNER JOIN  kettle.etl_createtable t2 ON t1.sub_id=t2.tablename #AND t1.conn_name=t2.`conn_name`;

DROP TABLE ccf_first_round_user_shop_splits;
CREATE TABLE ccf_first_round_user_shop_splits
AS
SELECT
    t.*,
    seq,
    SUBSTRING_INDEX(
        SUBSTRING_INDEX(t.`wifi_infos`, ';', seq),
        ';' ,- 1
    ) wifi_splits
FROM   sequence
CROSS JOIN (SELECT * FROM `ccf_first_round_user_shop_behavior` q ) t
WHERE seq BETWEEN 1 AND (SELECT 1 + LENGTH(t.`wifi_infos`) - LENGTH(REPLACE(t.`wifi_infos`, ';', '')));

drop table ccf_first_round_user_shop_detail;
CREATE TABLE ccf_first_round_user_shop_detail
AS
SELECT t.*, 
SUBSTRING_INDEX(SUBSTRING_INDEX(t.`wifi_splits`, '|', 1),'|' ,- 1)  bssid,
SUBSTRING_INDEX(SUBSTRING_INDEX(t.`wifi_splits`, '|', 2),'|' ,- 1)  signa,
SUBSTRING_INDEX(SUBSTRING_INDEX(t.`wifi_splits`, '|', 3),'|' ,- 1)  flag
FROM ccf_first_round_user_shop_splits t;



SELECT COUNT(*),COUNT(DISTINCT shop_id),COUNT(DISTINCT category_id),COUNT(DISTINCT mall_id) FROM ccf_first_round_shop_info;
/*
count(*)  count(distinct shop_id) count(distinct category_id) COUNT(DISTINCT mall_id)
8477  8477  67  97
*/

SELECT COUNT(*),COUNT(DISTINCT user_id),COUNT(DISTINCT shop_id) FROM ccf_first_round_user_shop_behavior;
/*
COUNT(*)  COUNT(DISTINCT user_id) COUNT(DISTINCT shop_id)
1138015 714608  8477
*/

SELECT COUNT(*),COUNT(DISTINCT user_id),COUNT(DISTINCT mall_id) FROM evaluation_public;
/*
COUNT(*)	COUNT(DISTINCT user_id)	COUNT(DISTINCT mall_id)
483931	338642	97
*/

SELECT MAX(LENGTH(wifi_infos)) FROM `ccf_first_round_user_shop_behavior`;--419

SELECT * FROM ccf_first_round_user_shop_behavior;

SELECT MIN(signa),MAX(signa) FROM ccf_first_round_user_shop_detail;
-- -1 -99

SELECT COUNT(*),COUNT(DISTINCT longitude,latitude),COUNT(DISTINCT shop_id) FROM `ccf_first_round_shop_info`
-- 8477 7189 8477

SELECT COUNT(*)
FROM
(SELECT DISTINCT user_id FROM `evaluation_public`) t1
LEFT JOIN (SELECT DISTINCT user_id FROM `ccf_first_round_user_shop_behavior`) t2
ON t1.user_id=t2.user_id
WHERE t2.user_id IS NULL; --247307

SELECT COUNT(DISTINCT user_id) FROM `evaluation_public`;--338642

SELECT COUNT(*)
FROM
(SELECT DISTINCT longitude,latitude FROM `evaluation_public`) t1
LEFT JOIN (SELECT DISTINCT longitude,latitude FROM `ccf_first_round_user_shop_behavior`) t2
ON t1.longitude=t2.longitude AND t1.latitude=t2.latitude
WHERE t2.longitude IS NULL;--356735

SELECT COUNT(DISTINCT longitude,latitude) FROM `evaluation_public`;--407607

SELECT COUNT(*)
FROM 
(SELECT DISTINCT shop_id FROM ccf_first_round_user_shop_behavior) t1
LEFT JOIN (SELECT DISTINCT shop_id,mall_id FROM ccf_first_round_shop_info) t2
ON t1.shop_id=t2.shop_id
WHERE t2.shop_id IS NULL; --0

SELECT COUNT(*)
FROM
(SELECT DISTINCT mall_id FROM `evaluation_public`) t1
LEFT JOIN ccf_first_round_shop_info t2
ON t1.mall_id=t2.mall_id
WHERE t2.mall_id IS NULL; --0

SELECT MIN(price),MAX(price) FROM ccf_first_round_shop_info ;
-- 24 88 

SELECT * FROM ccf_first_round_user_shop_detail WHERE user_id='u_376';


ALTER TABLE `tianchi`.`ccf_first_round_shop_info`   
  ADD  INDEX `idx_shop_id` (`shop_id`);

ALTER TABLE `tianchi`.`ccf_first_round_user_shop_behavior`   
  ADD  INDEX `idx_shop_id` (`shop_id`);
ALTER TABLE `tianchi`.`ccf_first_round_shop_info`   
  ADD  INDEX `idx_mall_id` (`mall_id`);

CREATE TABLE ccf_first_round_user_shop_distance
AS
SELECT t1.user_id
,t1.time_stamp 
,t1.longitude AS shop_longitude
,t1.latitude AS shop_latitude
,t3.shop_id
,t3.category_id
,t3.longitude
,t3.latitude
,t3.price
,t3.mall_id
,SQRT(POWER(t1.longitude-t3.longitude,2)+POWER(t1.latitude-t3.latitude,2)) AS distance
,CASE WHEN t1.shop_id=t3.shop_id THEN 1 ELSE 0 END AS target
FROM ccf_first_round_user_shop_behavior t1
LEFT JOIN ccf_first_round_shop_info t2 ON t1.shop_id=t2.shop_id
LEFT JOIN ccf_first_round_shop_info t3 ON t2.mall_id=t3.mall_id;

CREATE TABLE ccf_first_round_user_shop_distance1
AS
SELECT t1.user_id
,t1.time_stamp 
,t1.longitude AS shop_longitude
,t1.latitude AS shop_latitude
,t2.shop_id
,t2.mall_id
FROM ccf_first_round_user_shop_behavior t1
LEFT JOIN ccf_first_round_shop_info t2 ON t1.shop_id=t2.shop_id
--WHERE t1.user_id='u_376';

CREATE TABLE ccf_first_round_user_shop_distance2
AS
SELECT t2.user_id
,t2.time_stamp 
,t2.shop_longitude
,t2.shop_latitude
,t3.shop_id
,t3.category_id
,t3.longitude
,t3.latitude
,t3.price
,t3.mall_id
,SQRT(POWER(t2.shop_longitude-t3.longitude,2)+POWER(t2.shop_latitude-t3.latitude,2)) AS distance
,CASE WHEN t2.shop_id=t3.shop_id THEN 1 ELSE 0 END AS target
FROM ccf_first_round_user_shop_distance1 t2
LEFT JOIN ccf_first_round_shop_info t3 ON t2.mall_id=t3.mall_id;
--105749945 行受到影响

SELECT target,COUNT(*) FROM ccf_first_round_user_shop_distance2 GROUP BY target;
target  COUNT(*)
0 104611930
1 1138015


SELECT MAX(distance) FROM ccf_first_round_user_shop_distance2 WHERE target =1 ;
MAX(distance)
231.64104876287462

SELECT COUNT(*),COUNT(DISTINCT mall_id),COUNT(DISTINCT user_id),SUM(CASE WHEN target=1 THEN 1 ELSE 0 END) AS t FROM ccf_first_round_user_shop_distance2 WHERE distance >1

COUNT(*)  COUNT(DISTINCT mall_id) COUNT(DISTINCT user_id) t
75103 63  627 832

SELECT MAX(distance) FROM ccf_first_round_user_shop_distance2 WHERE target =1 AND distance <1;
MAX(distance)
0.9617772343952651

CREATE TABLE ccf_first_round_user_shop_straightdistance
AS
SELECT 
t1.`shop_id`
,t1.`category_id`
,t1.`price`
,t1.`mall_id`
,t1.longitude AS shop_longitude
,t1.latitude AS shop_latitude
,t2.longitude
,t2.latitude
,SQRT(  
    (  
     ((t1.longitude-t2.longitude)*PI()*12656*COS(((t1.latitude+t2.latitude)/2)*PI()/180)/180)  
     *  
     ((t1.longitude-t2.longitude)*PI()*12656*COS (((t1.latitude+t2.latitude)/2)*PI()/180)/180)  
    )  
    +  
    (  
     ((t1.latitude-t2.latitude)*PI()*12656/180)  
     *  
     ((t1.latitude-t2.latitude)*PI()*12656/180)  
    ))  distance
FROM
  `ccf_first_round_user_shop_behavior`  t2
LEFT JOIN
 `ccf_first_round_shop_info`   t1
ON t1.shop_id=t2.shop_id;

SELECT 
t1.`shop_id`
,t1.`category_id`
,t1.`price`
,t1.`mall_id`
,t1.longitude AS shop_longitude
,t1.latitude AS shop_latitude
,t2.longitude
,t2.latitude
,SQRT(  
    (  
     ((t1.longitude-t2.longitude)*PI()*12656*COS(((t1.latitude+t2.latitude)/2)*PI()/180)/180)  
     *  
     ((t1.longitude-t2.longitude)*PI()*12656*COS (((t1.latitude+t2.latitude)/2)*PI()/180)/180)  
    )  
    +  
    (  
     ((t1.latitude-t2.latitude)*PI()*12656/180)  
     *  
     ((t1.latitude-t2.latitude)*PI()*12656/180)  
    ))  distance1
, ACOS(
 SIN((t1.latitude*PI())/180) * SIN((t2.latitude*PI())/180) + 
 COS((t1.latitude*PI())/180) * COS((t2.latitude*PI())/180) * COS((t1.longitude*PI())/180 - (t2.longitude*PI())/180)
 )*6370.996 AS distance2
,ROUND(6378.138*2*ASIN(SQRT(POW(SIN( (t1.latitude*PI()/180-t2.latitude*PI()/180)/2),2)+COS(t1.latitude*PI()/180)*COS(t2.latitude*PI()/180)* POW(SIN( (t1.longitude*PI()/180-t2.longitude*PI()/180)/2),2)))*1000,2) AS distance3
FROM
  `ccf_first_round_user_shop_behavior`  t2
LEFT JOIN
 `ccf_first_round_shop_info`   t1
ON t1.shop_id=t2.shop_id;

DROP TABLE IF EXISTS ccf_first_round_user_shop_straightdistance;
CREATE TABLE ccf_first_round_user_shop_straightdistance
AS
SELECT 
t2.user_id
,t1.`shop_id`
,t1.`category_id`
,t1.`price`
,t1.`mall_id`
,t1.longitude AS shop_longitude
,t1.latitude AS shop_latitude
,t2.longitude
,t2.latitude
,ROUND(6378.138*2*ASIN(SQRT(POW(SIN( (t1.latitude*PI()/180-t2.latitude*PI()/180)/2),2)+COS(t1.latitude*PI()/180)*COS(t2.latitude*PI()/180)* POW(SIN( (t1.longitude*PI()/180-t2.longitude*PI()/180)/2),2)))*1000,2) AS distance
FROM
  `ccf_first_round_user_shop_behavior`  t2
LEFT JOIN
 `ccf_first_round_shop_info`   t1
ON t1.shop_id=t2.shop_id;


DROP TABLE if exists ccf_first_round_user_shop_straightdistance_agg;
CREATE TABLE ccf_first_round_user_shop_straightdistance_agg
AS
SELECT shop_id,category_id,mall_id,COUNT(*),MIN(distance),MAX(distance),AVG(distance) 
FROM ccf_first_round_user_shop_straightdistance 
GROUP BY shop_id,category_id,mall_id;

SELECT mall_id,COUNT(*),MIN(distance),MAX(distance),AVG(distance) 
FROM ccf_first_round_user_shop_straightdistance 
GROUP BY mall_id;

SELECT COUNT(*),COUNT(DISTINCT user_id),COUNT(DISTINCT shop_id),COUNT(DISTINCT mall_id) FROM ccf_first_round_user_shop_behavior;

drop table if exists  ccf_first_round_user_shop_distance_kurt;
CREATE TABLE ccf_first_round_user_shop_distance_kurt
AS
SELECT t1.`shop_id`
,t1.`category_id`
,t1.`price`
,t1.`mall_id`
,t1.`shop_latitude`
,t1.`shop_longitude`
,t1.`latitude`
,t1.`longitude`
,t1.`distance`
,t2.`cnt`
,t2.`min_distance`
,t2.`max_distance`
,t2.`avg_distance`
,POWER(t1.`distance`-t2.`avg_distance`,4) AS k4
,POWER(t1.`distance`-t2.`avg_distance`,2) AS k2
FROM ccf_first_round_user_shop_straightdistance t1
LEFT JOIN ccf_first_round_user_shop_straightdistance_agg t2 ON t1.`shop_id`=t2.`shop_id`;

SELECT * FROM ccf_first_round_user_shop_straightdistance_agg;

SELECT * FROM ccf_first_round_user_shop_straightdistance;


SELECT POWER(3,4) FROM DUAL;

DROP TABLE if exists  ccf_first_round_user_shop_distance_kurt_shop;
CREATE TABLE ccf_first_round_user_shop_distance_kurt_shop
AS
SELECT shop_id,AVG(k4) AS k,POWER(AVG(k2),2) AS s,AVG(k4)*1.0/POWER(AVG(k2),2) AS kurt
FROM ccf_first_round_user_shop_distance_kurt
GROUP BY shop_id;

DROP TABLE if exists ccf_first_round_user_shop_distance_kurt_category;
CREATE TABLE ccf_first_round_user_shop_distance_kurt_category
AS
SELECT category_id,AVG(k4) AS k,POWER(AVG(k2),2) AS s,AVG(k4)*1.0/POWER(AVG(k2),2) AS kurt
FROM ccf_first_round_user_shop_distance_kurt
GROUP BY category_id;

DROP TABLE if exists ccf_first_round_user_shop_distance_kurt_mall;
CREATE TABLE ccf_first_round_user_shop_distance_kurt_mall
AS
SELECT mall_id,AVG(k4) AS k,POWER(AVG(k2),2) AS s,AVG(k4)*1.0/POWER(AVG(k2),2) AS kurt
FROM ccf_first_round_user_shop_distance_kurt
GROUP BY mall_id;

SELECT * FROM ccf_first_round_user_shop_distance_kurt_mall;


SELECT COUNT(*),COUNT(DISTINCT shop_id),COUNT(DISTINCT bssid) FROM `ccf_first_round_user_shop_detail`
COUNT(*)  COUNT(DISTINCT shop_id) COUNT(DISTINCT bssid)
11075599  8477  399679

CREATE TABLE ccf_first_round_user_shop_detail_agg
AS
SELECT shop_id,bssid,COUNT(*) cnt FROM `ccf_first_round_user_shop_detail` GROUP BY shop_id,bssid;
