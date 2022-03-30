# -*- coding: utf-8 -*-
import random
import math


def annealingoptimize(domain, costf, T=10000.0, cool=0.95, step=1):
    """模拟退火算法搜索最优解"""
    # 随机初始化值
    vec = [float(random.randint(domain[i][0], domain[i][1]))
           for i in range(len(domain))]
    while T > 0.1:
        # 选择一个索引值
        i = random.randint(0, len(domain) - 1)
        # 选择一个改变索引值的方向
        dir = random.randint(-step, step)
        # 创建一个代表题解的一个新列表，改变其中一个值
        vecb = vec[:]
        vecb[i] += dir
        if vecb[i] < domain[i][0]:
            vecb[i] = domain[i][0]
        elif vecb[i] > domain[i][1]:
            vecb[i] = domain[i][1]
        # 计算当前成本和新的成本
        ea = costf(vec)
        eb = costf(vecb)
        # 它是更好解吗，或者是趋于最优解的可能的临界解吗？
        if (eb < ea or random.random() < pow(math.e, -(eb - ea) / T)):
            vec = vecb
        # 降低温度
        T = T * cool
    return vec
