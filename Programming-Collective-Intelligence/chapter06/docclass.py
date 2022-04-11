import re
import math


def getwords(doc):
    """���ı��н���������ȡ"""
    splitter = re.compile('\\W*')
    # ���ݷ���ĸ�ַ����е��ʲ��
    words = [s.lower() for s in splitter.split(doc)
             if 2 < len(s) < 20]
    # ֻ����һ�鲻�ظ��ĵ���
    return dict([(w, 1) for w in words])


class classifier:
    def __init__(self, getfeatures, filename=None):
        # ͳ������/������ϵ�����
        self.fc = {}
        # ͳ��ÿ�������е��ĵ�����
        self.cc = {}
        self.getfeatures = getfeatures

    def incf(self, f, cat):
        """���Ӷ�����/������ϵļ���ֵ"""
        self.fc.setdefault(f, {})
        self.fc[f].setdefault(cat, 0)
        self.fc[f][cat] += 1

    def incc(self, cat):
        """���Ӷ�ĳһ����ļ���ֵ"""
        self.cc.setdefault(cat, 0)
        self.cc[cat] += 1

    def fcount(self, f, cat):
        """ĳһ����������ĳһ�����еĴ���"""
        if f in self.fc and cat in self.fc[f]:
            return float(self.fc[f][cat])
        return 0.0

    def catcount(self, cat):
        """����ĳһ�����е�����������"""
        if cat in self.cc:
            return float(self.cc[cat])
        return 0

    def totalcount(self):
        """���������������"""
        return sum(self.cc.values())

    def categories(self):
        """���з�����б�"""
        return self.cc.keys()
