import datetime
import os

import numpy as np
import pandas as pd
import re
import string
from pandas import read_parquet
import math

from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
from collections import Counter
# 平时使用
# import hanlp
# 仅获取新词时使用
from pyhanlp import *

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
os.environ['JAVA_HOME'] = '/home/zhuzhicheng/java/jdk-20/'
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 设置环境变量


HEADLINE_LEN = 100
ABSTRACT_LEN = 500


# 用于扩充字典的文件
def find_newword(in_path, out_path):
    print(1)
    df = read_parquet('./filter/vnews_stock_split.parquet')
    # 限定长度
    df = df[df['date'].str.contains('2022')][:100000]
    res = []

    for index, item in df.iterrows():
        res.append(item['content'])
    print('sentences nums: {}'.format(len(res)))
    # 打开文件以写入模式
    file = open(in_path, 'w')
    for text in res:
        file.write(text + '\n')
    file.close()
    del res
    print('finding new wordsnums')

    f = open(in_path, mode="r", encoding="utf-8")
    data = f.read()
    f.close()
    word_info_list = HanLP.extractWords(data, 10000, True)  # 第2个参数为提取的词语数量，第3个参数为是否过滤词典中已有的词语

    file = open(out_path, 'w')

    for text in word_info_list:
        file.write(str(text) + '\n')
    file.close()


def add_newword():
    CustomDictionary = JClass("com.hankcs.hanlp.dictionary.CustomDictionary")
    fileHandler = open('./word_split/output.txt', 'r', encoding='utf-8')
    while True:
        # Get next line from file
        line = fileHandler.readline()
        #  line = line.decode('utf-8')
        # If line is empty then end of file reached
        if not line:
            break
        #        print(line.strip())
        CustomDictionary.add(line.strip())
        # Close Close
    fileHandler.close()


def hanlp2word(df):
    # tok.load

    pos = list(string.punctuation)
    pos.extend(["！", "？", "。", "，", "；", "：", "（", "）", "【", "】", "、", "《", "》", "‘", "’", "“", "”"])
    stopword_list = [sw.replace('\n', '') for sw in open('./word_split/stopwords.txt', encoding='utf-8').readlines()]
    pos.extend(stopword_list)
    ta = time.time()

    words = df['content'].to_list()
    res = []
    print('data length {}'.format(len(df)))

    for w in tqdm(words):
        tmp = list(set(t.word for t in HanLP.segment(w)))
        res.extend(tmp)

    def deal_pos(data):
        # 遍历字典的键，如果键在标点符号列表中，则删除该键值对
        for p in pos:
            data.pop(p, None)
        return data

    res = Counter(res)
    res = deal_pos(res)
    vis = {k: v for k, v in res.items() if len(k) > 1 and v >= 5}
    vis = sorted(vis.items(), key=lambda x: -x[1])[:1000]
    vis = [k for k, v in vis]

    pattern = re.compile(r'\d+')
    my_list = [item for item in vis if not pattern.search(item)]

    tb = time.time()
    print('split word cost time {}'.format(int(tb - ta)))

    with open("./word_split/output_hanlp.txt", "w") as f:
        # 将 list 中的元素逐个写入文件
        for item in my_list[:800]:
            f.write("%s\n" % item)


def get_vector(df_path, path, mode):
    df = read_parquet(df_path)

    # 限定词频为600维度
    notice_word = set(
        [sw.replace('\n', '') for sw in open(path, encoding='utf-8').readlines()][:600])
    # notice_word = set(
    #     [sw.replace('\n', '') for sw in open('./market/output_chatglm.txt', encoding='utf-8').readlines()])
    cnt = len(notice_word)

    #  出现该词的文档数量,文档总数
    show = [0] * cnt
    num = len(df)

    # 计算词频特征
    def fun1(s):
        nonlocal show
        res = [0] * cnt
        for index, word in enumerate(notice_word):
            if word in s:
                res[index] += 1
        show = list(map(lambda x: 1 + x[1] if x[0] > 0 else x[1], zip(res, show)))
        return res

    # 计算tf-idf
    def fun2(s):
        nonlocal show
        for index, word in enumerate(s):
            s[index] = s[index] * (math.log((1 + num) / (1 + show[index])) + 1)
        res = sum([x ** 2 for x in s]) ** 0.5
        if res > 0:
            s = list(map(lambda x: round(x / res, 2), s))

        return s

    # 合并同一股票的tf-idf
    def fun3(s):
        # 使用 多个值取max
        multiple_lists = [t for t in s]
        arrays = [np.array(x) for x in multiple_lists]
        res = np.array([max(k) for k in zip(*arrays)])

        return res

    # 单文档词频出现数
    df['frquent'] = df['content'].apply(lambda x: fun1(x))
    print('fruqent 1 is finish')
    # tf-idf 计算值
    df['frquent'] = df['frquent'].apply(lambda x: fun2(x))
    print('fruqent 2 is finish')
    # 将同一天的股票新闻与机构合并
    df1 = df.groupby(['date', 'stock_id'])['frquent'].apply(lambda grp: fun3(grp)).reset_index()
    df2 = df.groupby(['date', 'stock_id'])['ref_pct'].apply(lambda grp: list(grp)[0]).reset_index()
    res = pd.merge(df1, df2)
    # del df1, df2

    # 过滤较少数据的项目
    df3 = df.groupby(['date', 'stock_id'])['pct'].apply(lambda grp: list(grp)).reset_index()
    df3['len'] = df3['pct'].apply(lambda x: len(x))
    df3['pct'] = df3['pct'].apply(lambda x: x[0])
    df3 = df3[df3['len'] >= 3].reset_index()
    res = pd.merge(res, df3)

    print(len(res))
    res = res[['date', 'frquent', 'stock_id', 'ref_pct', 'pct']]

    def judge(x, a, b):
        if x <= a:
            return 0
        elif x <= b:
            return 1
        else:
            return 2

    q10 = res.quantile(.3, numeric_only=True).ref_pct
    q90 = res.quantile(.7, numeric_only=True).ref_pct
    res['label'] = res['ref_pct'].apply(lambda x: judge(x, q10, q90))
    res['label'].astype('int')
    lbl = preprocessing.LabelEncoder()
    res['label'] = lbl.fit_transform(res['label'].astype(str))  # 将提示的包含错误数据类型这一列进行转换

    data_x = np.array(res['frquent'].values.tolist())
    data_y = res['label'].values.tolist()
    if mode == 'train':
        return data_x, data_y
    else:
        data_id = res['stock_id'].values.tolist()
        return data_x,data_id

    # hanlp
    # res.to_parquet('filter/vnews_stock_merge2.parquet')

    # chatglm
    # df.to_parquet('filter/vnews_stock_merge3.parquet')


if __name__ == "__main__":
    # 必要扩充此表数据
    find_newword('./word_split/input.txt', './word_split/output.txt')
    # data = read_parquet('./filter/vnews_stock_split.parquet')
    # data = data[data['date'].str.contains('2022-01')]
    # # hanlp2word(data)
    # data = get_vector(data, './word_split/output_hanlp.txt')
    # print(data.head())
