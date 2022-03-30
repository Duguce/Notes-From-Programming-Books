# -*- coding: utf-8 -*-
import time
import random
import math
from randomoptimize import randomoptimize
from hillclimb import hillclimb

people = [('Seymour', 'BOS'),
          ('Franny', 'DAL'),
          ('Zooey', 'CAK'),
          ('Walt', 'MIA'),
          ('Buddy', 'ORD'),
          ('Les', 'OMA')]
# New York 的 LaGuardia 机场
destination = 'LGA'

flights = {}
for line in open('schedule.txt'):
    origin, dest, depart, arrive, price = line.strip().split(',')
    flights.setdefault((origin, dest), [])
    # 将航班信息添加到航班列表中
    flights[(origin, dest)].append((depart, arrive, int(price)))


def getminutes(t):
    """计算某个给定时间在一天中的分钟数"""
    x = time.strptime(t, '%H:%M')
    return x[3] * 60 + x[4]


def printschedule(r):
    """将人们决定搭乘的所有航班打印成列表"""
    for d in range(len(r) // 2):
        name = people[d][0]
        origin = people[d][1]
        out = flights[(origin, destination)][int(r[2 * d])]
        ret = flights[(destination, origin)][int(r[2 * d + 1])]
        print('%10s%10s %5s-%5s $%3s %5s-%5s $%3s' % (name, origin,
                                                      out[0], out[1], out[2],
                                                      ret[0], ret[1], ret[2]))


def schedulecost(sol):
    """总的旅行成本以及不同家庭成员在机场总的等待时间"""
    totalprice = 0
    latestarrival = 0
    earliestdep = 24 * 60
    for d in range(len(sol) // 2):
        # 得到往程航班和返程航班
        origin = people[d][1]
        outbound = flights[(origin, destination)][int(sol[2 * d])]
        returnf = flights[(destination, origin)][int(sol[2 * d + 1])]
        # 总价格等于所有往程航班和返程航班价格之和
        totalprice += outbound[2]
        totalprice += returnf[2]
        # 记录最晚到达时间和最早离开时间
        latestarrival = max(getminutes(outbound[1]), latestarrival)
        earliestdep = min(getminutes(returnf[0]), earliestdep)
    # 每个人必须在机场等待直到最后一个人到达为止
    # 每个人也必须在相同时间到达，并等候他们的返程航班
    totalwait = 0
    for d in range(len(sol) // 2):
        origin = people[d][1]
        outbound = flights[(origin, destination)][int(sol[2 * d])]
        returnf = flights[(destination, origin)][int(sol[2 * d + 1])]
        totalwait += latestarrival - getminutes(outbound[1])
        totalwait += getminutes(returnf[0]) - earliestdep
    # 这个题解要求多付一天的汽车租用费用吗？如果是，则费用为50美元
    if latestarrival > earliestdep: totalprice += 50
    return totalprice + totalwait


if __name__ == '__main__':
    domain = [(0, 9)] * (len(people) * 2)
    # s = randomoptimize(domain, schedulecost)
    s = hillclimb(domain, schedulecost)
    print(schedulecost(s))
    printschedule(s)
