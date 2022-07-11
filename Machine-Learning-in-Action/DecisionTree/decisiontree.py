# -*- coding: UTF-8 -*
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from math import log
import operator
import pickle


def calcShannonEnt(dataSet):
    """
    计算给定数据集的香农熵
    :param dataSet:数据集
    :return:shannonEnt - 香农熵
    """
    numEntries = len(dataSet)  # 返回数据集的行数
    labelCounts = {}  # 保存每一个标签（label）出现次数
    for featVec in dataSet:  # 对每组特征向量进行统计
        currentLabel = featVec[-1]  # 提取标签（label）信息
        if currentLabel not in labelCounts.keys():  # 如果标签（label）没有放入统计次数的字典，添加进去
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  # label计数
    shannonEnt = 0.0  # 香农熵
    for key in labelCounts:  # 计算香农熵
        prob = float(labelCounts[key]) / numEntries  # 选择该标签（label）的概率
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt  # 返回经验熵


def creatDataSet():
    """
    创建测试数据集
    :return:
    dataSet - 数据集
    labels - 特征标签
    """
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    """
    按照给定特征划分数据集
    :param dataSet:待划分的数据集
    :param axis:划分数据集的特征
    :param value:需要返回的特征的值
    :return:无
    """
    retDataSet = []  # 创建返回的数据集列表
    for featVec in dataSet:  # 遍历数据集
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # 去掉axis特征
            reducedFeatVec.extend(featVec[axis + 1:])  # 将符合条件的添加到返回的数据集
            retDataSet.append(reducedFeatVec)
    return retDataSet  # 返回划分后的数据集


def chooseBestFeatureToSplit(dataSet):
    """
    选择最好的数据集划分方式
    :param dataSet:数据集
    :return:bestFeature - 香农熵增益最大的（最优）特征的索引值
    """
    numFeatures = len(dataSet[0]) - 1  # 特征数量
    baseEntropy = calcShannonEnt(dataSet)  # 计算数据集的香农熵
    bestInfoGain = 0.0  # 信息增益
    bestFeature = -1  # 最优特征的索引值
    for i in range(numFeatures):  # 遍历所有特征
        # 获取dataSet的第i个所有特征
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  # 创建set集合{} 元素不可重复
        newEntropy = 0.0  # 经验条件熵
        for value in uniqueVals:  # 计算信息增益
            subDataSet = splitDataSet(dataSet, i, value)  # subDataSet划分后的子集
            prob = len(subDataSet) / float(len(dataSet))  # 计算子集的概率
            newEntropy += prob * calcShannonEnt(subDataSet)  # 根据公式计算经验条件尚
        infoGain = baseEntropy - newEntropy  # 信息增益
        if (infoGain > bestInfoGain):  # 计算信息增益
            bestInfoGain = infoGain  # 更新信息增益，找到最大的信息增益
            bestFeature = i  # 记录信息增益最大的特征的索引值
    return bestFeature  # 返回信息增益最大的特征的索引值


def majorityCnt(classList):
    """
    统计classList中出现此处最多的元素（类标签）
    :param classList: 类标签列表
    :return: sortedClassCount[0][0] - 出现此处最多的元素（类标签）
    """
    classCount = {}  # 统计classList中每个元素出现的次数
    for vote in classList:
        if vote not in classCount.key(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 根据字典的值降序排序
    return sortedClassCount[0][0]


def createTree(dataSet, labels, featLabels):
    """
    创建决策树
    :param dataSet:训练数据集
    :param labels: 分类属性标签
    :param featLabels: 存储选择的最优特征标签
    :return: myTree - 决策树
    """
    classList = [example[-1] for example in dataSet]  # 取分类标签
    if classList.count(classList[0]) == len(classList):  # 如果类别完全相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1 or len(labels) == 0:  # 遍历完所有特征时返回出现次数最多的类标签
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 选择最优特征
    bestFeatLabel = labels[bestFeat]  # 最优特征的标签
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel: {}}  # 根据最优特征的标签生成树
    del (labels[bestFeat])  # 删除已经使用特征标签
    featValues = [example[bestFeat] for example in dataSet]  # 得到训练集中所有最优特征的属性值
    uniqueVals = set(featValues)  # 去掉重复的属性值
    for value in uniqueVals:  # 遍历特征，创建决策树
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels, featLabels)

    return myTree


def getNumLeafs(myTree):
    """
    获取决策树叶子结点的数目
    :param myTree: 决策树
    :return: numLeafs - 决策树的叶子结点的数目
    """
    numLeafs = 0  # 初始化叶子结点
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]  # 获取下一组字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    """
    获取决策树的层数
    :param myTree:决策树
    :return:maxDepth - 决策树的层数
    """
    maxDepth = 0  # 初始化决策树深度
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]  # 获取下一个字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth  # 更新层数
    return maxDepth


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """
    绘制节点
    :param nodeTxt:节点名
    :param centerPt: 文本位置
    :param parentPt: 标注的箭头位置
    :param nodeType: 节点格式
    :return: 无
    """
    arrow_args = dict(arrowstyle="<-")  # 定义箭头格式
    # font = FontProperties(fname=r"c:\windows\fonts\simsunb.ttf", size=14)  # 设置中文字体
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',  # 绘制结点
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def plotMidText(cntrPt, parentPt, txtString):
    """
    标注有向边属性值
    :param cntrPt:用于计算标注位置
    :param parentPt:用于计算标注位置
    :param txtString:标注的内容
    :return:
    """
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]  # 计算标注位置
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
    """
    绘制决策树
    :param myTree:决策树
    :param parentPt:标注的内容
    :param nodeTxt:结点名
    :return:
    """
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")  # 设置结点格式
    leafNode = dict(boxstyle="round4", fc="0.8")  # 设置叶结点格式
    numLeafs = getNumLeafs(myTree)  # 获取决策树叶结点数目，决定了树的宽度
    depth = getTreeDepth(myTree)  # 获取决策树层数
    firstStr = next(iter(myTree))  # 下个字典
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)  # 中心位置
    plotMidText(cntrPt, parentPt, nodeTxt)  # 标注有向边属性值
    plotNode(firstStr, cntrPt, parentPt, decisionNode)  # 绘制结点
    secondDict = myTree[firstStr]  # 下一个字典，也就是继续绘制子结点
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD  # y偏移
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            plotTree(secondDict[key], cntrPt, str(key))  # 不是叶结点，递归调用继续绘制
        else:  # 如果是叶结点，绘制叶结点，并标注有向边属性值
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    """
    创建绘制面板
    :param inTree:决策树
    :return:无
    """
    fig = plt.figure(1, facecolor='white')  # 创建fig
    fig.clf()  # 清空fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # 去掉x、y轴
    plotTree.totalW = float(getNumLeafs(inTree))  # 获取决策树叶结点数目
    plotTree.totalD = float(getTreeDepth(inTree))  # 获取决策树层数
    plotTree.xOff = -0.5 / plotTree.totalW;
    plotTree.yOff = 1.0;  # x偏移
    plotTree(inTree, (0.5, 1.0), '')  # 绘制决策树
    plt.show()


def classify(inputTree, featLabels, testVec):
    """
    使用决策树分类
    :param inputTree:已经生成的决策树
    :param featLabels:存储选择的最优特征标签
    :param testVec:测试数据列表，顺序对应最优特征标签
    :return:classLabel - 分类结果
    """
    firstStr = next(iter(inputTree))  # 获取决策树结点
    secondDict = inputTree[firstStr]  # 下一个字典
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree, filename):
    """
    存储决策树
    :param inputTree:已经生成的决策树
    :param filename:决策树的存储文件名
    :return:无
    """
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)


def grabTree(filename):
    """
    读取决策树
    :param filename: 决策树的存储文件名
    :return: pickle.load(fr) - 决策树字典
    """
    fr = open(filename, 'rb')
    return pickle.load(fr)


if __name__ == '__main__':
    dataSet, labels = creatDataSet()
    featLabels = []
    myTree = createTree(dataSet, labels, featLabels)
    createPlot(myTree)
    testVec = [0, 1]  # 测试数据
    result = classify(myTree, featLabels, testVec)
    if result == 'yes':
        print('鱼类')
    if result == 'no':
        print('非鱼类')
