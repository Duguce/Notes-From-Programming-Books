# -*- coding: utf-8 -*-
import recommendations


def loadMovieLens(path='../data/movielens'):
    # 获取影片标题
    movies = {}
    for line in open(f'{path}/u.item', "r", encoding='ISO-8859-1'):
        (id, title) = line.split("|")[:2]
        movies[id] = title
    # 加载数据
    prefs = {}
    for line in open(f'{path}/u.data', "r", encoding='ISO-8859-1'):
        (user, movieid, rating, ts) = line.split("\t")
        prefs.setdefault(user, {})
        prefs[user][movies[movieid]] = float(rating)
    return prefs


def main():
    prefs = loadMovieLens()
    print(recommendations.getRecommendations(prefs, '87')[0:30])
    itemsim = recommendations.calculateSimilarItems(prefs, n=50)
    print(recommendations.getRecommendedItems(prefs, itemsim, '87')[0:30])


if __name__ == '__main__':
    main()
