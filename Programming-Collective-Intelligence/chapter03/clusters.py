# -*- coding: utf-8 -*-
from math import sqrt
from PIL import Image, ImageDraw
import random


def readfile(filename):
    lines = list(open(filename))
    # 第一行是列标题
    colnames = lines[0].strip().split('\t')[1:]
    rownames = []
    data = []
    for line in lines[1:]:
        p = line.strip().split('\t')
        # 每一行的第一列是行名
        rownames.append(p[0])
        # 剩余部分就是该行对应的数据
        data.append([float(x) for x in p[1:]])
    return rownames, colnames, data


def pearson(v1, v2):
    """皮尔逊相似度计算"""
    # 简单求和
    sum1 = sum(v1)
    sum2 = sum(v2)
    # 求平方和
    sum1Sq = sum(pow(v, 2) for v in v1)
    sum2Sq = sum(pow(v, 2) for v in v2)
    # 求乘积之和
    pSum = sum(v1[i] * v2[i] for i in range(len(v1)))
    # 计算r （Pearson score）
    num = pSum - (sum1 * sum2 / len(v1))
    den = sqrt((sum1Sq - pow(sum1, 2) / len(v1)) * (sum2Sq - pow(sum2, 2) / len(v1)))
    if den == 0: return 0
    return 1.0 - num / den


def tanimoto(v1, v2):
    """Tanimoto计算距离"""
    c1, c2, shr = 0, 0, 0
    for i in range(len(v1)):
        if v1[i] != 0: c1 += 1  # 出现在v1中
        if v2[i] != 0: c2 += 1  # 出现在v2中
        if v1[i] != 0 and v2[i] != 0: shr += 1  # 在两个向量中都出现

    return 1.0 - (float(shr) / (c1 + c2 - shr + 1))  # 在两个向量中都出现


class bicluster:
    """聚类"""

    def __init__(self, vec, left=None, right=None, distance=0.0, id=None):
        self.left = left
        self.right = right
        self.vec = vec
        self.id = id
        self.distance = distance


def hcluster(rows, distance=pearson):
    distances = {}
    currentclustid = -1
    # 最开始的聚类就是数据集中的行
    clust = [bicluster(rows[i], id=i) for i in range(len(rows))]
    while len(clust) > 1:
        lowestpair = (0, 1)
        closest = distance(clust[0].vec, clust[1].vec)
        # 遍历每一个配对，寻找最小距离
        for i in range(len(clust)):
            for j in range(i + 1, len(clust)):
                # 用distances来缓存距离的计算值
                if (clust[i].id, clust[j].id) not in distances:
                    distances[(clust[i].id, clust[j].id)] = distance(clust[i].vec, clust[j].vec)
                d = distances[(clust[i].id, clust[j].id)]
                if d < closest:
                    closest = d
                    lowestpair = (i, j)
        # 计算两个聚类的平均值
        mergevec = [(clust[lowestpair[0]].vec[i] + clust[lowestpair[1]].vec[i]) / 2.0 for i in
                    range(len(clust[0].vec))]
        # 建立新的聚类
        newcluster = bicluster(mergevec, left=clust[lowestpair[0]], right=clust[lowestpair[1]], distance=closest,
                               id=currentclustid)
        # 不在原始集合中的聚类，其id为负数
        currentclustid -= 1
        del clust[lowestpair[1]]
        del clust[lowestpair[0]]
        clust.append(newcluster)
    return clust[0]


def kcluster(rows, distance=pearson, k=4):
    """K-均值聚类算法"""
    # 确定每个点的最小值和最大值
    ranges = [
        (min(row[i] for row in rows), max(row[i] for row in rows))
        for i in range(len(rows[0]))
    ]
    # 随机创建k个中心点
    clusters = [[
        random.random() * (ranges[i][1] - ranges[i][0]) + ranges[i][0]
        for i in range(len(rows[0]))
    ] for j in range(k)]
    lastmatches = None
    for t in range(100):
        print('Iteration %d' % t)
        bestmatches = [[] for i in range(k)]
        # 在每一行中寻找距离最近的中心点
        for j in range(len(rows)):
            row = rows[j]
            bestmatch = 0
            for i in range(k):
                d = distance(clusters[i], row)
                if d < distance(clusters[bestmatch], row): bestmatch = i
            bestmatches[bestmatch].append(j)
        # 如果结果与上一次相同，则整个过程结束
        if bestmatches == lastmatches: break
        lastmatches = bestmatches
        # 把中心点移到其所有成员的平均位置处
        for i in range(k):
            avgs = [0.0] * len(rows[0])
            if len(bestmatches[i]) > 0:
                for rowid in bestmatches[i]:
                    for m in range(len(rows[rowid])):
                        avgs[m] += rows[rowid][m]
                for j in range(len(avgs)):
                    avgs[j] /= len(bestmatches[i])
                clusters[i] = avgs
    return bestmatches


def printclust(clust, labels=None, n=0):
    """递归遍历聚类数，并将其以类似文件系统层级结构的形式打印出来"""
    # 利用缩进来建立层级布局
    for _ in range(n):
        print(' ', end=" ")
    if clust.id < 0:
        # 负数标记代表这是一个分支
        print('-')
    elif labels is None:
        print(clust.id)
    else:
        # 正数标记代表一个叶节点
        print(labels[clust.id])
    # 现在开始打印右侧分支和左侧分支
    if clust.left != None: printclust(clust.left, labels=labels, n=n + 1)
    if clust.right != None: printclust(clust.right, labels=labels, n=n + 1)


def getheight(clust):
    """生成给定聚类的总体高度"""
    # 这是一个叶节点吗？若是，则高度为1
    if clust.left == None and clust.right == None: return 1
    # 否则，高度为每个分支的高度之和
    return getheight(clust.left) + getheight(clust.right)


def getdepth(clust):
    """获取根节点的总体误差,并生成一个缩放因子（scaling factor）"""
    # 一个叶节点的距离是0.0
    if clust.left == None and clust.right == None: return 0
    # 一个枝节点的距离等于左右两侧分支中距离较大者
    # 加上该枝节点自身的距离
    return max(getdepth(clust.left), getdepth(clust.right)) + clust.distance


def drawdendrogram(clust, labels, jpeg='clusters.jpg'):
    # 高度和宽度
    h = getheight(clust) * 20
    w = 1200
    depth = getdepth(clust)
    # 由于宽度是固定的，因此我们须要对距离值做出相应的调整
    scaling = float(w - 150) / depth
    # 新建一个白色背景的图片
    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.line((0, h / 2, 10, h / 2), fill=(255, 0, 0))
    # 画出第一个节点
    drawnode(draw, clust, 10, (h / 2), scaling, labels)
    img.save(jpeg, 'JPEG')


def drawnode(draw, clust, x, y, scaling, labels):
    """接受一个聚类及其位置作为输入参数，计算节点的位置，并用线条进行连接"""
    if clust.id < 0:
        h1 = getheight(clust.left) * 20
        h2 = getheight(clust.right) * 20
        top = y - (h1 + h2) / 2
        bottom = y + (h1 + h2) / 2
        # 线的长度
        l1 = clust.distance * scaling
        # 聚类到其子节点的垂直线
        draw.line((x, top + h1 / 2, x, bottom - h2 / 2), fill=(255, 0, 0))
        # 连接左侧节点的水平线
        draw.line((x, top + h1 / 2, x + 11, top + h1 / 2), fill=(255, 0, 0))
        # 连接右侧节点的水平线
        draw.line((x, bottom - h2 / 2, x + 11, bottom - h2 / 2), fill=(255, 0, 0))
        # 调用函数绘制左右节点
        drawnode(draw, clust.left, x + 11, top + h1 / 2, scaling, labels)
        drawnode(draw, clust.right, x + 11, bottom - h2 / 2, scaling, labels)
    else:
        # 如果这是一个叶节点，则绘制节点的标签
        draw.text((x + 5, y - 7), labels[clust.id], (0, 0, 0))


def rotatematrix(data):
    newdata = []
    for i in range(len(data[0])):
        newrow = [data[j][i] for j in range(len(data))]
        newdata.append(newrow)
    return newdata


def scaledown(data, distance=pearson, rate=0.01):
    """计算各项之间的距离"""
    n = len(data)
    # 每一对数据项之间的真实距离
    realdist = [[distance(data[i], data[j]) for j in range(n)] for i in range(n)]
    outersum = 0.0
    # 随机初始化节点在二维空间中的起始位置
    loc = [[random.random(), random.random()] for i in range(n)]
    fakedist = [[0.0 for j in range(n)] for i in range(n)]
    lasterror = None
    for m in range(0, 1000):
        # 寻找投影后的距离
        for i in range(n):
            for j in range(n):
                fakedist[i][j] = sqrt(sum([pow(loc[i][x] - loc[j][x], 2) for x in range(len(loc[i]))]))
        # 移动节点
        grad = [[0.0, 0.0] for i in range(n)]
        totalerror = 0
        for k in range(n):
            for j in range(n):
                if j == k: continue
                # 误差值等于目标距离与当前距离之间差值的百分比
                errorterm = (fakedist[j][k] - realdist[j][k]) / realdist[j][k]
                # 每一个节点都需要根据误差的多少按照比例移离或移向其他节点
                grad[k][0] += ((loc[k][0] - realdist[j][0]) / fakedist[j][k]) * errorterm
                grad[k][1] += ((loc[k][1] - realdist[j][0]) / fakedist[j][k]) * errorterm
                # 记录总的误差值
                totalerror += abs(errorterm)
        print(totalerror)
        # 如果节点移动之后的情况变得更糟，则程序结束
        if lasterror and lasterror < totalerror: break
        lasterror = totalerror
        # 根据rate参数与grad值相乘的结果，移动每一个节点
        for k in range(n):
            loc[k][0] -= rate * grad[k][0]
            loc[k][1] -= rate * grad[k][1]
    return loc


def draw2d(data, labels, jpeg='mds2d.jpg'):
    """根据坐标值，在二维图上展示坐标的位置及其对应的标签"""
    img = Image.new('RGB', (2000, 2000), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i in range(len(data)):
        x = (data[i][0] + 0.5) * 1000
        y = (data[i][1] + 0.5) * 1000
        draw.text((x, y), labels[i], (0, 0, 0))
    img.save(jpeg, 'JPEG')


def main():
    blognames, words, data = readfile('blogdata.txt')
    # clust = hcluster(data)
    # printclust(clust, labels=blognames)
    # drawdendrogram(clust, blognames, jpeg='blogclust.jpg')
    # rdata = rotatematrix(data)
    # wordclust = hcluster(rdata)
    # drawdendrogram(wordclust, labels=words, jpeg='wordclust.jpg')
    # kclust = kcluster(data, k=10)
    # print([blognames[r] for r in kclust[0]])
    # print([blognames[r] for r in kclust[1]])
    # wants, people, data = readfile('zebo.txt')
    # clust = hcluster(data, distance=tanimoto)
    # drawdendrogram(clust, wants)
    coords = scaledown(data)
    draw2d(coords, blognames, jpeg='blogs2d.jpg')


if __name__ == '__main__':
    main()
