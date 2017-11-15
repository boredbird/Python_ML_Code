#!/usr/bin/python
# -*- coding:utf-8 -*-

import operator
# operator.__mul__(a, b)
# Return a * b, for a and b numbers.

def c(n, k):
    # 求组合数
    return reduce(operator.mul, range(n-k+1, n+1)) / reduce(operator.mul, range(1, k+1))

def bagging(n, p):
    s = 0
    for i in range(n / 2 + 1, n + 1):
        s += c(n, i) * p ** i * (1 - p) ** (n - i)
    return s

if __name__ == "__main__":
    for t in range(9, 100, 10):
        # 假设事件发生的概率为0.6
        print t, '次采样正确率：', bagging(t, 0.6)
