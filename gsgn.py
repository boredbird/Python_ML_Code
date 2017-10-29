# -*- coding:utf-8 -*-
__author__ = 'maomaochong'
import pandas as pd
gsgn = pd.read_csv(r'E:\gsgn.csv')

var = gsgn.iloc[0,].values[0]


def get_ip(var):
    start_str = var[0].split('-')[0]
    end_str = var[0].split('-')[1]
    start = str(int(start_str[0:2], 16)) \
            + '.' + str(int(start_str[2:4], 16)) \
            + '.' + str(int(start_str[4:6], 16)) \
            + '.' + str(int(start_str[6:], 16))

    end = str(int(end_str[0:2], 16)) \
            + '.' + str(int(end_str[2:4], 16)) \
            + '.' + str(int(end_str[4:6], 16)) \
            + '.' + str(int(end_str[6:], 16))

    return start,end


# start,end = gsgn.apply(get_ip,axis=1)

start = []
end = []
for var in gsgn.values:
    tmp_start,tmp_end = get_ip(var)
    start.append(tmp_start)
    end.append(tmp_end)


result = pd.DataFrame({'SGSNNET':gsgn['SGSNNET']
                        ,'start':start
                        ,'end':end})
result.to_csv(r'E:\gsgn_result.csv',index=None)