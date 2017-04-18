# -*- coding:utf-8 -*-
__author__ = 'chunhui.zhang'

# adict = locals()
# lst1 =  [1,2,3,4,5,6,7,8,9]
# for i,s in enumerate(lst1):
#     adict['a%s' % (i+1)] =s
#
# print a1,a2,a3,a4,a5,a6,a7,a8,a9


adict = locals()
lst1 =  ['a','b','c']
for i,s in enumerate(lst1):
    print i,s
    adict['a%s' % (s)] =i

print aa