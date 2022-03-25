# -*- coding: utf-8 -*-
import feedparser
import re


def getwordcounts(url):
    """返回一个RSS订阅源的标题和包含单词计数情况的字典"""
    # 解析订阅源
    d = feedparser.parse(url)
    wc = {}
    # 循环遍历所有的文章条目
    for e in d.entries:
        summary = e.summary if 'summary' in e else e.description
        # 提取一个单词列表
        words = getwords(f'{e.title}{summary}')
        for word in words:
            wc.setdefault(word, 0)
            wc[word] += 1
    return getattr(d.feed, 'title', 'Unknown title'), wc


def getwords(html):
    # 去除所有HTML标记
    txt = re.compile(r'<[^>]+>').sub('', html)
    # 利用所有非字母字符拆分出单词
    words = re.compile(r'[^A-Z^a-z]+').split(txt)
    # 转化成小写形式
    return [word.lower() for word in words if word != '']


def main():
    apcount = {}
    wordcounts = {}
    feedlist = [line for line in open('feedlist.txt')]
    for feedurl in feedlist:
        title, wc = getwordcounts(feedurl)
        wordcounts[title] = wc
        for word, count in wc.items():
            apcount.setdefault(word, 0)
            if count > 1:
                apcount[word] += 1
    wordlist = []
    for w, bc in apcount.items():
        frac = float(bc) / len(feedlist)
        if frac > 0.1 and frac < 0.5: wordlist.append(w)
    out = open('blogdata.txt', 'w')
    out.write('Blog')
    for word in wordlist: out.write('\t%s' % word)
    out.write('\n')
    for blog, wc in wordcounts.items():
        out.write(blog)
        for word in wordlist:
            if word in wc:
                out.write('\t%d' % wc[word])
            else:
                out.write('\t0')
        out.write('\n')


if __name__ == '__main__':
    main()
