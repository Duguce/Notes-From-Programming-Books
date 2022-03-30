# -*- coding: utf-8 -*-
import random


def hillclimb(domain, costf):
    """爬山法搜索最优解"""
    # 创建一个随机解
    sol = [random.randint(domain[i][0], domain[i][1])
           for i in range(len(domain))]
    # 主循环
    while 1:
        # 创建相邻解的列表
        neighbors = []
        for j in range(len(domain)):
            # 在每个方向上相对于原值偏离一点
            if sol[j] > domain[j][0]:
                neighbors.append(sol[:j] + [sol[j] - 1] + sol[j + 1:])
            if sol[j] < domain[j][1]:
                neighbors.append(sol[:j] + [sol[j] + 1] + sol[j + 1:])
        # 在相邻解中寻找最优解
        current = costf(sol)
        best = current
        for neighbor in neighbors:
            cost = costf(neighbor)
            if cost < best:
                best = cost
                sol = neighbor
        # 如果没有更好的解，则退出循环
        if best == current:
            break
    return sol
