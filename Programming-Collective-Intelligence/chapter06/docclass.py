import re
import math


def getwords(doc):
    """从文本中进行特征提取"""
    splitter = re.compile('\\W*')
    # 根据非字母字符进行单词拆分
    words = [s.lower() for s in splitter.split(doc)
             if 2 < len(s) < 20]
    # 只返回一组不重复的单词
    return dict([(w, 1) for w in words])


class classifier:
    def __init__(self, getfeatures, filename=None):
        # 统计特征/分类组合的数量
        self.fc = {}
        # 统计每个分类中的文档数量
        self.cc = {}
        self.getfeatures = getfeatures

    def incf(self, f, cat):
        """增加对特征/分类组合的计数值"""
        self.fc.setdefault(f, {})
        self.fc[f].setdefault(cat, 0)
        self.fc[f][cat] += 1

    def incc(self, cat):
        """增加对某一分类的计数值"""
        self.cc.setdefault(cat, 0)
        self.cc[cat] += 1

    def fcount(self, f, cat):
        """某一特征出现于某一分类中的次数"""
        if f in self.fc and cat in self.fc[f]:
            return float(self.fc[f][cat])
        return 0.0

    def catcount(self, cat):
        """属于某一分类中的内容项数量"""
        if cat in self.cc:
            return float(self.cc[cat])
        return 0

    def totalcount(self):
        """所有内容项的数量"""
        return sum(self.cc.values())

    def categories(self):
        """所有分类的列表"""
        return self.cc.keys()
