 ##用户属性表
 CREATE TABLE user_01
AS
SELECT
t1.`user_id`
,CASE WHEN t1.`age`='15岁以下' THEN 'a'
	WHEN t1.`age`='16-25岁' THEN 'b'
	WHEN t1.`age`='26-35岁' THEN 'c'
	WHEN t1.`age`='36-45岁' THEN 'd'
	WHEN t1.`age`='46-55岁' THEN 'e'
	WHEN t1.`age`='56岁以上' THEN 'f'
	ELSE 'g'
	END AS age
,t1.`sex`
,t1.`user_lv_cd`
,TO_DAYS('2016-04-10')-TO_DAYS(t1.`user_reg_dt`) AS rg_day_diff
FROM jdata_user t1;

select count(*) from `dmr`.`jdata_action_201604`;--13199934

SELECT COUNT(*) FROM `dmr`.`jdata_action_201603`;--25916378

SELECT COUNT(*) FROM `dmr`.`jdata_action_201602`;--11485424

CREATE TABLE jdata_action_all
AS
SELECT *
FROM
(SELECT * FROM `jdata_action_201602` 
UNION ALL 
SELECT * FROM `jdata_action_201603` 
UNION ALL 
SELECT * FROM `jdata_action_201604`
) t;

CREATE TABLE user_02
AS
SELECT
user_id
,SUM(CASE WHEN TYPE=1 THEN 1 ELSE 0 END) AS browse_cnt
,SUM(CASE WHEN TYPE=2 THEN 1 ELSE 0 END) AS shoppingcar_in_cnt
,SUM(CASE WHEN TYPE=3 THEN 1 ELSE 0 END) AS shoppingcar_out_cnt
,SUM(CASE WHEN TYPE=4 THEN 1 ELSE 0 END) AS order_cnt
,SUM(CASE WHEN TYPE=5 THEN 1 ELSE 0 END) AS follow_cnt
,SUM(CASE WHEN TYPE=6 THEN 1 ELSE 0 END) AS click_cnt
FROM jdata_action_all t
where time<'2016-04-11' 
group by user_id;

CREATE TABLE user_03
AS
select
t1.user_id
,t1.age
,t1.sex
,t1.user_lv_cd
,t1.rg_day_diff
,t2.browse_cnt
,t2.shoppingcar_in_cnt
,t2.shoppingcar_out_cnt
,t2.order_cnt
,t2.follow_cnt
,t2.order_cnt/t2.browse_cnt  as browse_order_rate
,t2.order_cnt/t2.shoppingcar_in_cnt  as shopcar_order_rate
,t2.order_cnt/t2.follow_cnt  as follow_order_rate
,t2.order_cnt/t2.click_cnt  as click_order_rate
from user_01 t1
left join user_02 t2
on t1.user_id=t2.user_id;


SELECT * FROM jdata_action_201602;
 
 ##商品属性表
 create table dim_sku
as
select sku_id,cate,brand
FROM jdata_action_all t
group by 1;
 
 CREATE TABLE product_01
AS
SELECT
sku_id
,SUM(CASE WHEN TYPE=1 THEN 1 ELSE 0 END) AS browse_cnt
,SUM(CASE WHEN TYPE=2 THEN 1 ELSE 0 END) AS shoppingcar_in_cnt
,SUM(CASE WHEN TYPE=3 THEN 1 ELSE 0 END) AS shoppingcar_out_cnt
,SUM(CASE WHEN TYPE=4 THEN 1 ELSE 0 END) AS order_cnt
,SUM(CASE WHEN TYPE=5 THEN 1 ELSE 0 END) AS follow_cnt
,SUM(CASE WHEN TYPE=6 THEN 1 ELSE 0 END) AS click_cnt
FROM jdata_action_all t
where time<'2016-04-11' 
group by sku_id;

 SELECT 
sku_id
attr1
attr2
attr3
cate
brand
FROM `jdata_product`

SELECT 
sku_id
,comment_num
,has_bad_comment
,bad_comment_rate
FROM  `jdata_comment`
WHERE dt ='2016-04-04'
 
create table product_02
as
select
t1.sku_id
,t1.attr1
,t1.attr2
,t1.attr3
,t1.cate
,t1.brand
,t2.comment_num
,t2.has_bad_comment
,t2.bad_comment_rate
,t3.browse_cnt
,t3.shoppingcar_in_cnt
,t3.shoppingcar_out_cnt
,t3.order_cnt
,t3.follow_cnt
,t3.order_cnt*1.0/t3.browse_cnt  as browse_order_rate
,t3.order_cnt*1.0/t3.shoppingcar_in_cnt  as shopcar_order_rate
,t3.order_cnt*1.0/t3.follow_cnt  as follow_order_rate
,t3.order_cnt*1.0/t3.click_cnt  as click_order_rate
from jdata_product t1
left join jdata_comment t2 on t1.sku_id=t2.sku_id and t2.dt ='2016-04-04'
left join product_01 t3 on t1.sku_id=t3.sku_id ;
 
 
 ##用户行为数据
 
 ##品类属性
  CREATE TABLE cate_01
AS
SELECT
cate
,SUM(CASE WHEN TYPE=1 THEN 1 ELSE 0 END) AS browse_cnt
,SUM(CASE WHEN TYPE=2 THEN 1 ELSE 0 END) AS shoppingcar_in_cnt
,SUM(CASE WHEN TYPE=3 THEN 1 ELSE 0 END) AS shoppingcar_out_cnt
,SUM(CASE WHEN TYPE=4 THEN 1 ELSE 0 END) AS order_cnt
,SUM(CASE WHEN TYPE=5 THEN 1 ELSE 0 END) AS follow_cnt
,SUM(CASE WHEN TYPE=6 THEN 1 ELSE 0 END) AS click_cnt
FROM jdata_action_all t
where time<'2016-04-11' 
group by cate;
 

 ##品牌属性
  CREATE TABLE brand_01
AS
SELECT
brand
,SUM(CASE WHEN TYPE=1 THEN 1 ELSE 0 END) AS browse_cnt
,SUM(CASE WHEN TYPE=2 THEN 1 ELSE 0 END) AS shoppingcar_in_cnt
,SUM(CASE WHEN TYPE=3 THEN 1 ELSE 0 END) AS shoppingcar_out_cnt
,SUM(CASE WHEN TYPE=4 THEN 1 ELSE 0 END) AS order_cnt
,SUM(CASE WHEN TYPE=5 THEN 1 ELSE 0 END) AS follow_cnt
,SUM(CASE WHEN TYPE=6 THEN 1 ELSE 0 END) AS click_cnt
FROM jdata_action_all t
where time<'2016-04-11' 
group by brand;

drop table cate_02;
CREATE TABLE cate_02
AS
SELECT
cate
,t2.browse_cnt
,t2.shoppingcar_in_cnt
,t2.shoppingcar_out_cnt
,t2.order_cnt
,t2.follow_cnt
,t2.order_cnt*1.0/t2.browse_cnt  AS browse_order_rate
,t2.order_cnt*1.0/t2.shoppingcar_in_cnt  AS shopcar_order_rate
,t2.order_cnt*1.0/t2.follow_cnt  AS follow_order_rate
,t2.order_cnt*1.0/t2.click_cnt  AS click_order_rate
FROM cate_01 t2;

drop table brand_02;
CREATE TABLE brand_02
AS
SELECT
brand
,t2.browse_cnt
,t2.shoppingcar_in_cnt
,t2.shoppingcar_out_cnt
,t2.order_cnt
,t2.follow_cnt
,t2.order_cnt*1.0/t2.browse_cnt  AS browse_order_rate
,t2.order_cnt*1.0/t2.shoppingcar_in_cnt  AS shopcar_order_rate
,t2.order_cnt*1.0/t2.follow_cnt  AS follow_order_rate
,t2.order_cnt*1.0/t2.click_cnt  AS click_order_rate
FROM brand_01 t2;

CREATE TABLE brand_03
AS
SELECT t2.`brand`
,COUNT(DISTINCT t2.`sku_id`) AS sku_cnt
,AVG(t1.`comment_num`) AS comment_num_avg
,AVG(t1.`has_bad_comment`) AS has_bad_comment_avg
,AVG(t1.bad_comment_rate) AS bad_comment_rate_avg
,SUM(t1.`comment_num`) AS comment_num_sum
,SUM(t1.`has_bad_comment`) AS has_bad_comment_sum
,SUM(t1.bad_comment_rate) AS bad_comment_rate_sum
,SUM(t1.`has_bad_comment`)*1.0/COUNT(DISTINCT t2.`sku_id`) AS has_bad_comment_sku_rate
FROM `dim_sku` t2
LEFT JOIN jdata_comment t1
ON t1.`sku_id`=t2.`sku_id`
WHERE t1.dt ='2016-04-04'
GROUP BY t2.brand;

SELECT COUNT(*),COUNT(DISTINCT brand) FROM brand_02;

create table brand_04
as
select
t1.*
,sku_cnt
,comment_num_avg
,has_bad_comment_avg
,bad_comment_rate_avg
,comment_num_sum
,has_bad_comment_sum
,bad_comment_rate_sum
,has_bad_comment_sku_rate
from brand_02 t1
left join brand_03 t2
on t1.brand=t2.brand;
 
 
CREATE TABLE cate_brand_sku_action_cnt
AS
SELECT 
t1.sku_id
,t1.cate
,t1.brand
,SUM(CASE WHEN t2.type=1 THEN 1 ELSE 0 END) AS browse_cnt
,SUM(CASE WHEN t2.type=2 THEN 1 ELSE 0 END) AS shoppingcar_in_cnt
,SUM(CASE WHEN t2.type=3 THEN 1 ELSE 0 END) AS shoppingcar_out_cnt
,SUM(CASE WHEN t2.type=4 THEN 1 ELSE 0 END) AS order_cnt
,SUM(CASE WHEN t2.type=5 THEN 1 ELSE 0 END) AS follow_cnt
,SUM(CASE WHEN t2.type=6 THEN 1 ELSE 0 END) AS click_cnt
FROM dim_sku t1
LEFT JOIN `jdata_action_all` t2 ON t1.sku_id=t2.sku_id and t2.time<'2016-04-11' 
GROUP BY t1.sku_id,t1.cate,t1.brand;


CREATE TABLE cate_brand_sku_action_cnt_01
AS
SELECT cate
,brand
,SUM(browse_cnt) AS browse_cnt_sum
,SUM(shoppingcar_in_cnt) AS shoppingcar_in_cnt_sum
,SUM(shoppingcar_out_cnt) AS shoppingcar_out_cnt_sum
,SUM(order_cnt) AS order_cnt_sum
,SUM(follow_cnt) AS follow_cnt_sum
,SUM(click_cnt) AS click_cnt_sum
 FROM cate_brand_sku_action_cnt 
 GROUP BY cate
,brand;

CREATE TABLE cate_brand_action_rank_browse
AS
SELECT b.*
,@rownum:=@rownum+1 
,IF(@pdept=b.cate,@rank:=@rank+1,@rank:=1) AS browse_cnt_sum_rank
,@pdept:=b.cate
FROM 
(
SELECT *
 FROM cate_brand_sku_action_cnt_01 
 ORDER BY cate,browse_cnt_sum DESC
) b 
,(SELECT @rownum :=0 , @pdept := NULL ,@rank:=0) c ;

CREATE TABLE cate_brand_action_rank_shoppingcar_in
AS
SELECT b.cate,b.brand
,@rownum:=@rownum+1 
,IF(@pdept=b.cate,@rank:=@rank+1,@rank:=1) AS shoppingcar_in_cnt_sum_rank
,@pdept:=b.cate
FROM 
(
SELECT *
 FROM cate_brand_sku_action_cnt_01 
 ORDER BY cate,shoppingcar_in_cnt_sum DESC
) b 
,(SELECT @rownum :=0 , @pdept := NULL ,@rank:=0) c ;


CREATE TABLE cate_brand_action_rank_shoppingcar_out
AS
SELECT b.cate,b.brand
,@rownum:=@rownum+1 
,IF(@pdept=b.cate,@rank:=@rank+1,@rank:=1) AS shoppingcar_out_cnt_sum_rank
,@pdept:=b.cate
FROM 
(
SELECT *
 FROM cate_brand_sku_action_cnt_01 
 ORDER BY cate,shoppingcar_out_cnt_sum DESC
) b 
,(SELECT @rownum :=0 , @pdept := NULL ,@rank:=0) c ;


CREATE TABLE cate_brand_action_rank_order
AS
SELECT b.cate,b.brand
,@rownum:=@rownum+1 
,IF(@pdept=b.cate,@rank:=@rank+1,@rank:=1) AS order_cnt_sum_rank
,@pdept:=b.cate
FROM 
(
SELECT *
 FROM cate_brand_sku_action_cnt_01 
 ORDER BY order_cnt_sum DESC
) b 
,(SELECT @rownum :=0 , @pdept := NULL ,@rank:=0) c ;

CREATE TABLE cate_brand_action_rank_follow
AS
SELECT b.cate,b.brand
,@rownum:=@rownum+1 
,IF(@pdept=b.cate,@rank:=@rank+1,@rank:=1) AS follow_cnt_sum_rank
,@pdept:=b.cate
FROM 
(
SELECT *
 FROM cate_brand_sku_action_cnt_01 
 ORDER BY follow_cnt_sum DESC
) b 
,(SELECT @rownum :=0 , @pdept := NULL ,@rank:=0) c ;

CREATE TABLE cate_brand_action_rank_click
AS
SELECT b.cate,b.brand
,@rownum:=@rownum+1 
,IF(@pdept=b.cate,@rank:=@rank+1,@rank:=1) AS click_cnt_sum_rank
,@pdept:=b.cate
FROM 
(
SELECT *
 FROM cate_brand_sku_action_cnt_01 
 ORDER BY click_cnt_sum DESC
) b 
,(SELECT @rownum :=0 , @pdept := NULL ,@rank:=0) c ;

CREATE TABLE inspector_01
AS
SELECT user_id
,sku_id
,MAX(TO_DAYS(TIME))
,MIN(TO_DAYS(TIME))
,MAX(TO_DAYS(TIME))-MIN(TO_DAYS(TIME)) AS time_diff 
FROM jdata_action_all
GROUP BY user_id,sku_id;


SELECT time_diff,COUNT(*) 
FROM inspector_01
GROUP BY time_diff;


SELECT COUNT(*) FROM jdata_action_all WHERE TYPE=4;--48252

CREATE TABLE inspector_02
AS
SELECT * FROM jdata_action_all WHERE TYPE=4;

SELECT SUM(cnt)
FROM 
(SELECT user_id,sku_id,COUNT(*) AS cnt FROM inspector_02 GROUP BY user_id,sku_id 
) t
WHERE t.cnt>1;--2853 重复下单

SELECT t2.*
FROM 
(SELECT user_id,sku_id,COUNT(*) AS cnt 
FROM inspector_02 
GROUP BY user_id,sku_id 
) t1
INNER JOIN inspector_02 t2 ON t1.user_id=t2.user_id AND t1.sku_id=t2.sku_id
WHERE t1.cnt>1
ORDER BY t2.user_id,t2.sku_id;

#查看下单的用户时长，为数据集划分提供依据
CREATE TABLE inspector_03
AS
SELECT t1.`user_id`,t1.`sku_id`
,MAX(TO_DAYS(t2.TIME)) AS max_time
,MIN(TO_DAYS(t2.TIME)) AS min_time
,MAX(TO_DAYS(t2.TIME))-MIN(TO_DAYS(t2.TIME)) AS time_diff 
FROM inspector_02 t1
INNER JOIN jdata_action_all t2 
ON t1.user_id=t2.user_id AND t1.sku_id=t2.sku_id
GROUP BY t1.`user_id`,t1.`sku_id`;




####cate_brand_sku_comment
select count(*) from `trainset4_feature_interval_acc` where order_cnt>0;--12271

select count(*) from  `trainset4_lable`;--3133

select count(*)
from trainset4_lable t1
inner join trainset4_feature_interval_acc t2
on t1.user_id=t2.user_id and t1.sku_id=t2.sku_id;--1139


select count(*) from `trainset3_feature_interval_acc` where order_cnt>0;--22198

select count(*) from  `trainset3_lable`;--2801

select count(*)
from trainset3_lable t1
inner join trainset3_feature_interval_acc t2
on t1.user_id=t2.user_id and t1.sku_id=t2.sku_id;--1207

##有行为的用户下单率依然很低
##行为 194万 user_id,sku_id
##行为 91380 user_id
##行为期间下单 22198 user_id,sku_id
##预测期间下单 1207 user_id,sku_id
##应该人工干预，有效减少
#行为期间已经下过单的用户，因为复购占比不高
#行为期间未下单，预测期间也不下单的，这部分是大头
##时间分布：最后下单时间 与 最开始有行为时间的差值
##时间分布：最后下单时间 与 次级有行为时间的差值（为了过滤掉行为时间久远，不太可能下单的）


select day_cnt,count(*)
from
(select user_id,sku_id,datediff(time_x,time_y) as day_cnt
from `inspector_03` t
where t.type_x =4
and t.type_y <> 4
group by user_id,sku_id) tt
group by day_cnt;



select t.*,datediff(time_x,time_y) as day_cnt
from `inspector_03` t
where t.type_x =4
and t.type_y <> 4
order by day_cnt desc






