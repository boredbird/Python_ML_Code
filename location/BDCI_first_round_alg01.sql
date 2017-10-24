"""
直接根据ccf_first_round_shop_info表提供的每个shop_id的经纬度，
evaluation_public数据集匹配shop_info经纬度，找到最近的shop_id作为结果提交
"""

create table evaluation_public_shop_info
as
select 
t1.row_id
,t1.user_id
,t1.mall_id
,t1.longitude
,t1.latitude
,t2.shop_id
,t2.category_id
,t2.longitude as longitude_shop
,t2.latitude as latitude_shop
,t2.price
from evaluation_public t1
inner join ccf_first_round_shop_info t2 on t1.mall_id=t2.mall_id;

create table evaluation_public_shop_distance
as
select t1.*
,ROUND(6367*2*ASIN(SQRT(POW(SIN( (t1.latitude*PI()/180-t1.latitude_shop*PI()/180)/2),2)+COS(t1.latitude*PI()/180)*COS(t1.latitude_shop*PI()/180)* POW(SIN( (t1.longitude*PI()/180-t1.longitude_shop*PI()/180)/2),2)))*1000,2) AS distance
from evaluation_public_shop_info t1;

create table BDCI_first_round_alg01_submit
as
select t.row_id,t.shop_id 
from (select * FROM evaluation_public_shop_distance ORDER BY distance ASC) t
group by t.row_id,t.shop_id 
