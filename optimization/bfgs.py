# -*- coding:utf-8 -*-
__author__ = 'maomaochong'

from scipy.optimize import fmin,fmin_bfgs

def func(x):
    return x**2 - 4*x + 8

def afmin():
    x = [1.0]
    ans = fmin(func,x)
    print ans

def bfgs():
    x = [1.0]
    ans = fmin_bfgs(func,x)
    print ans

afmin()
bfgs()