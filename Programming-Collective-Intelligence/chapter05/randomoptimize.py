# -*- coding: utf-8 -*-
import random


def randomoptimize(domain, costf):
    """随机优化算法：随机搜索最优解"""
    best = 9999999
    bestr = None
    for i in range(10000):
        # 创建一个随机解
        r = [random.randint(domain[i][0], domain[i][1])
             for i in range(len(domain))]
        # 得到成本
        cost = costf(r)
        # 与目前为止的最优解进行比较
        if cost < best:
            best = cost
            bestr = r
    return r
