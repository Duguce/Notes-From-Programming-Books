# -*- coding: utf-8 -*-
from math import sqrt

# 一个涉及影评者及其对几部影片评分情况的字典
critics = {
    'Lisa Rose': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5, 'Just My Luck': 3.0, 'Superman Returns': 3.5,
                  'You, Me and Dupree': 2.5, 'The Night Listener': 3.0},
    'Gene Seymour': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5, 'Just My Luck': 1.5, 'Superman Returns': 5.0,
                     'You, Me and Dupree': 3.0, 'The Night Listener': 3.5},
    'Michael Phillips': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0, 'Superman Returns': 3.5,
                         'The Night Listener': 4.0},
    'Claudia Puig': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0, 'The Night Listener': 4.5, 'Superman Returns': 4.0,
                     'You, Me and Dupree': 2.0},
    'Mick LaSalle': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, 'Just My Luck': 2.0, 'Superman Returns': 3.0,
                     'The Night Listener': 3.0, 'You, Me and Dupree': 2.0},
    'Jack Matthews': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, 'The Night Listener': 3.0,
                      'Superman Returns': 5.0,
                      'You, Me and Dupree': 3.5},
    'Toby': {'Snakes on a Plane': 4.5, 'You, Me and Dupree': 1.0, 'Superman Returns': 4.0}}


def sim_distance(prefs, person1, person2):
    """
    返回一个有关person1和person2的基于距离的相似度评价
    得到shared_items的列表
    """
    si = {}
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item] = 1
    # 如果两者没有共同之处，则返回0
    if len(si) == 0: return 0
    # 计算所有差值的平方和
    sum_of_squares = sum(
        [pow(prefs[person1][item] - prefs[person2][item], 2) for item in prefs[person1] if item in prefs[person2]])
    return 1 / (1 + sqrt(sum_of_squares))


def sim_pearson(prefs, p1, p2):
    """
    返回p1和p2的皮尔逊相关系数
    得到双方都曾评价过的物品列表
    """
    si = {}
    for item in prefs[p1]:
        if item in prefs[p2]: si[item] = 1
    # 得到列表元素的个数
    n = len(si)
    # 如果两者没有共同之处，则返回1
    if n == 0: return 1
    # 对所有偏好求和
    sum1 = sum([prefs[p1][it] for it in si])
    sum2 = sum([prefs[p2][it] for it in si])
    # 求平方和
    sum1Sq = sum([pow(prefs[p1][it], 2) for it in si])
    sum2Sq = sum([pow(prefs[p2][it], 2) for it in si])
    # 求乘积之和
    pSum = sum([prefs[p1][it] * prefs[p2][it] for it in si])
    # 计算皮尔逊评价值
    num = pSum - (sum1 * sum2 / n)
    den = sqrt((sum1Sq - pow(sum1, 2) / n) * (sum2Sq - pow(sum2, 2) / n))
    if den == 0: return 0
    r = num / den
    return r


def sim_tonimoto(prefs, person1, person2):
    """tanimoto相似度计算"""
    si = {}
    # 判断有没有相同的数据，没有相同的数据则返回0
    for item in prefs[person1]:
        if item in prefs[person2]: si[item] = 1
    # 如果两者没有共同之处，则返回0
    if len(si) == 0: return 0
    si_num = len(si)
    person1_num = len(prefs[person1])
    person2_num = len(prefs[person2])
    res = float(si_num) / (person1_num + person2_num - si_num)
    return res


def topMatches(prefs, person, n=5, similarity=sim_pearson):
    """
    从反映偏好的字典中返回最为匹配者
    返回结果的个数和相似度函数均为可选参数
    """
    scores = [(similarity(prefs, person, other), other)
              for other in prefs if other != person]
    # 对列表进行排序，评价值最高者排在最前面
    scores.sort()
    scores.reverse()
    return scores[0:n]


def getRecommendations(prefs, person, similarity=sim_pearson):
    """利用所有他人评价值的加权平均，为某人提供建议"""
    totals = {}
    simSums = {}
    for other in prefs:
        # 不要和自己做比较
        if other == person: continue
        sim = similarity(prefs, person, other)
        # 忽略评价值为零或小于零的情况
        if sim <= 0: continue
        for item in prefs[other]:
            # 只对自己还未曾看到过的影片进行评价
            if item not in prefs[person] or prefs[person][item] == 0:
                # 相似度*评价值
                totals.setdefault(item, 0)
                totals[item] += prefs[other][item] * sim
                # 相似度之和
                simSums.setdefault(item, 0)
                # setdefault(keyname, value) keyname 您要从中返回值的项目的键名。 value 如果键存在，则此参数无效。如果键不存在，则此值将成为键的值。
                simSums[item] += sim
    # 建立一个归一化的列表
    rankings = [(total / simSums[item], item) for item, total in totals.items()]
    # 返回经过排序的列表
    rankings.sort()
    rankings.reverse()
    return rankings


def transformPrefs(prefs):
    result = {}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item, {})
            # 将物品和人员对调
            result[item][person] = prefs[person][item]
    return result


def calculateSimilarItems(prefs, n=10):
    """建立字典，以给出与这些物品最为相近的所有其他物品"""
    result = {}
    # 以物品为中心对偏好矩阵实施倒置处理
    itemPrefs = transformPrefs(prefs)
    c = 0
    for item in itemPrefs:
        # 针对大数据集更新状态变量
        c += 1
        if c % 100 == 0: print("%d / %d" % (c, len(itemPrefs)))
        # 寻找最相近的物品
        scores = topMatches(itemPrefs, item, n=n, similarity=sim_distance)
        result[item] = scores
    return result


def getRecommendedItems(prefs, itemMatch, user):
    userRatings = prefs[user]
    scores = {}
    totalSim = {}
    # 循环遍历由当前用户评分的物品
    for (item, rating) in userRatings.items():
        # 循环遍历与当前物品相近的物品
        for (similarity, item2) in itemMatch[item]:
            # 如果该用户已经对当前物品做过评价，则将其忽略
            if item2 in userRatings: continue
            # 评价值与相似度的加权之和
            scores.setdefault(item2, 0)
            scores[item2] += similarity * rating
            # 全部相似度之和
            totalSim.setdefault(item2, 0)
            totalSim[item2] += similarity
    # 将每个合计值除以加权和，求出平均值
    rankings = [(score / totalSim[item], item) for item, score in scores.items()]
    # 按照最高值到最低值的顺序，返回评分结果
    rankings.sort()
    rankings.reverse()
    return rankings


def main():
    print(sim_distance(critics, 'Lisa Rose', 'Gene Seymour'))
    print(sim_pearson(critics, 'Lisa Rose', 'Gene Seymour'))
    print(topMatches(critics, 'Toby', n=5))
    print(getRecommendations(critics, 'Toby'))
    print(getRecommendations(critics, 'Toby', similarity=sim_tonimoto))
    movies = transformPrefs(critics)
    print(topMatches(movies, 'Superman Returns'))
    itemsim = calculateSimilarItems(critics)
    print(itemsim)
    print(getRecommendedItems(critics, itemsim, 'Toby'))


if __name__ == '__main__':
    main()
