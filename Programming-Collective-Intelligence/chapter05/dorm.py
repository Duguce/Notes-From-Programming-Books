# -*- coding: utf-8 -*-
import random
import math
from randomoptimize import randomoptimize
from geneticoptimize import geneticoptimize
from hillclimb import hillclimb
from annealingoptimize import annealingoptimize


def printsolution(vec):
    """将学生和与之对应的宿舍打印输出"""
    slots = []
    # 为每个宿舍建两个槽
    for i in range(len(dorms)): slots += [i, i]
    # 遍历每一名同学的安置情况
    for i in range(len(vec)):
        x = int(vec[i])
        # 从剩余槽中选择
        dorm = dorms[slots[x]]
        # 输出学生及其被分配的宿舍
        print(prefs[i][0], dorm)
        # 删除该槽
        del slots[x]


def dormcost(vec):
    """成本函数"""
    cost = 0
    # 建立一个槽序列
    slots = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    # 遍历每一名学生
    for i in range(len(vec)):
        x = int(vec[i])
        dorm = dorms[slots[x]]
        pref = prefs[i][1]
        # 首选成本值为0，次选成本值为1
        if pref[0] == dorm:
            cost += 0
        elif pref[1] == dorm:
            cost += 1
        # 不在选择之列则成本值为3
        else:
            cost += 3
        # 删除选中的槽
        del slots[x]
    return cost


if __name__ == '__main__':
    # 代表宿舍，每个宿舍有两个可用的隔间
    dorms = ['Zeus', 'Athena', 'Hercules', 'Bacchus', 'Pluto']
    # 代表学生及其首选和次选
    prefs = [('Toby', ('Bacchus', 'Hercules')),
             ('Steve', ('Zeus', 'Pluto')),
             ('Andrea', ('Athena', 'Zeus')),
             ('Sarah', ('Zeus', 'Pluto')),
             ('Dave', ('Athena', 'Bacchus')),
             ('Jeff', ('Hercules', 'Pluto')),
             ('Fred', ('Pluto', 'Athena')),
             ('Suzie', ('Bacchus', 'Hercules')),
             ('Laura', ('Bacchus', 'Hercules')),
             ('Neil', ('Hercules', 'Athena'))]

    # [(0,9),(0,8),(0,7),(0,6),...,(0,0)]
    domain = [(0, (len(dorms) * 2) - i - 1) for i in range(len(dorms) * 2)]
    # printsolution([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    s = randomoptimize(domain, dormcost)
    dormcost(s)
    printsolution(s)
