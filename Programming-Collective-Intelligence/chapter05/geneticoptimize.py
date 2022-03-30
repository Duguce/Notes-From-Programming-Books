# -*- coding: utf-8 -*-
import random


def geneticoptimize(domain, costf, popsize=50, step=1, mutprob=0.2, elite=0.2, maxiter=100):
    """遗传算法搜索最优解"""

    # 变异操作
    def mutate(vec):
        i = random.randint(0, len(domain) - 1)
        if random.random() < 0.5 and vec[i] > domain[i][0]:
            return vec[0:i] + [vec[i] - step] + vec[i + 1:]
        elif vec[i] < domain[i][1]:
            return vec[0:i] + [vec[i] + step] + vec[i + 1:]

    # 交叉操作
    def crossover(r1, r2):
        i = random.randint(1, len(domain) - 2)
        return r1[:i] + r2[i:]

    # 构造初始种群
    pop = []
    for i in range(popsize):
        vec = [random.randint(domain[i][0], domain[i][1])
               for i in range(len(domain))]
        pop.append(vec)
    # 每一代有多少胜出者
    topelite = int(elite * popsize)
    # 主循环
    for i in range(maxiter):
        scores = [(costf(v), v) for v in pop]
        scores.sort()
        ranked = [v for (s, v) in scores]
        # 从纯粹的胜出者开始
        pop = ranked[0:topelite]
        # 添加变异和配对后的胜出者
        while len(pop) < popsize:
            if random.random() < mutprob:
                # 变异
                c = random.randint(0, topelite)
                pop.append(mutate(ranked[c]))
            else:
                # 交叉
                c1 = random.randint(0, topelite)
                c2 = random.randint(0, topelite)
                pop.append(crossover(ranked[c1], ranked[c2]))
        # 打印当前最优值
        print(scores[0][0])
    return scores[0][1]
