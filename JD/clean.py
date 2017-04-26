 # CREATE TABLE user_02
# AS
# SELECT
# user_id
# ,SUM(CASE WHEN TYPE=1 THEN 1 ELSE 0 END) AS browse_cnt
# ,SUM(CASE WHEN TYPE=2 THEN 1 ELSE 0 END) AS shoppingcar_in_cnt
# ,SUM(CASE WHEN TYPE=3 THEN 1 ELSE 0 END) AS shoppingcar_out_cnt
# ,SUM(CASE WHEN TYPE=4 THEN 1 ELSE 0 END) AS order_cnt
# ,SUM(CASE WHEN TYPE=5 THEN 1 ELSE 0 END) AS follow_cnt
# ,SUM(CASE WHEN TYPE=6 THEN 1 ELSE 0 END) AS click_cnt
# FROM jdata_action_all t
# where time<'2016-04-11' 
# group by user_id;


['browse_cnt','shoppingcar_in_cnt','shoppingcar_out_cnt','order_cnt','follow_cnt','click_cnt']


result01 = result[0].groupby(['user_id','sku_id'])['time_y'].agg({'time_y':np.max,'time_y':np.min})
{'time_y':np.max,'time_y':np.min})

dataset[0][dataset[0]['time'] < '2016-04-11'].groupby('user_id').agg()


dataset[0]['browse_cnt'] = dataset[0].groupby('user_id').apply(lambda x: x[x['type'] == 1].count())

g = dataset[0].groupby('user_id')
g.apply(lambda x: x[x['type'] == 1].count())





g = df.groupby('key1')
g.apply(lambda x: x[x['key2'] == 'one']['data1'].sum())


select a, b,
sum(case when d='xyz' then 1 else 0 end)/
count(distinct case when f='abc' then a end) as new_field
from table
group by a, b

df =pd.DataFrame({'A':[1,1,1,2,2,3,3,3], 'B':list('abcceegh'),'C':list('kmopmmmn'),'D':list('tauaaasf')})

def f(x):
    return x[x['C'] == 'm']['A'].sum()/ x[x['D'] == 'a']['A'].count()

df.groupby(['B']).apply(f)

df = pd.DataFrame({'key1' : ['a','a','b','b','a'],
               'key2' : ['one', 'two', 'one', 'two', 'one'],
               'data1' : np.random.randn(5),
               'data2' : np. random.randn(5)})
df
Out[7]: 
      data1     data2 key1 key2
0 -0.579223 -0.621723    a  one
1  0.534216  0.599761    a  two
2 -1.356058 -2.494947    b  one
3 -0.593667 -1.461809    b  two
4  0.866574  0.558372    a  one
g = df.groupby('key1')
g.apply(lambda x: x[x['key2'] == 'one']['data1'].sum())
Out[9]: 
key1
a    0.287352
b   -1.356058
dtype: float64


print df[(df['sex'] == 'Female') & (df['total_bill'] > 20)]

dataset[0][(dataset[0]['user_id'] == 200002) & (dataset[0]['type'] == 1)]


