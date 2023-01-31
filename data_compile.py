import datetime
import json
import time

import pandas as pd
import os

from pandas import read_parquet
from collections import defaultdict
from tqdm import tqdm

# os.environ['CUDA_VISIBLE_DEVICES'] = '5'

"""
这部分统一不同文件中的股票代码
"""


# 不同文件编码转化
def compile_id():
    df = pd.read_csv('./market/stock1.csv', encoding='gbk')

    with open('./market/id2stock.json', 'r+', encoding='utf-8') as file:
        content = file.read()
    content = json.loads(content)
    content = {v: k for k, v in content.items()}

    res = {}

    for index, row in df.iterrows():
        name, out_id = row['股票名'], row['股票代码']
        if name in content:
            res[out_id] = content[name]
    dict_json = json.dumps(res, ensure_ascii=False)

    with open('./market/id2id.json', 'w+', encoding='utf-8') as file:
        file.write(dict_json)


# 转换股票对应编码
def compile_stock():
    df = pd.read_csv('./market/stock.csv', encoding='gbk')

    with open('./market/id2stock.json', 'r+', encoding='utf-8') as file:
        content = file.read()
    content = json.loads(content)
    content = {v: k for k, v in content.items()}

    df['股票代码'] = df['股票名'].apply(lambda x: content[x])
    df.to_csv('./market/stock.csv', index=False, encoding='gbk')


"""
这部分处理前期数据，缩小数据体积
"""


def parquet_csv(path):
    df = read_parquet(path)
    filename = path.split('/')[-1]
    name = filename.split('.')[0]
    if name == 'md_security':
        df = df[['SECURITY_ID', 'SEC_FULL_NAME', 'SEC_SHORT_NAME', 'LIST_STATUS_CD']]
    elif name == 'vnews_nondupbd_flash':
        df = df[['NEWS_ID', 'NEWS_ORIGIN_SOURCE', 'NEWS_TITLE', 'NEWS_PUBLISH_SITE', 'EFFECTIVE_TIME']]
        df = df[['NEWS_ID', 'NEWS_TITLE', 'NEWS_PUBLISH_SITE', 'EFFECTIVE_TIME']]
        df.columns = ['NEWS_ID', 'NEWS_TITLE', 'NEWS_ORIGIN_SOURCE', 'EFFECTIVE_TIME']
        # df.columns = ['NEWS_ID', 'NEWS_TITLE', 'NEWS_ORIGIN_SOURCE', 'EFFECTIVE_TIME']
        df.fillna({'NEWS_ORIGIN_SOURCE': '参考消息'}, inplace=True)
    elif name == 'vnews_nondupbd_wechat':
        # df = df[['NEWS_ID', 'NEWS_ORIGIN_SOURCE', 'NEWS_TITLE', 'SITE_TYPE', 'EFFECTIVE_TIME']]
        df = df[['NEWS_ID', 'NEWS_ORIGIN_SOURCE', 'NEWS_TITLE', 'EFFECTIVE_TIME']]
        df.fillna({'NEWS_ORIGIN_SOURCE': '参考消息'}, inplace=True)
    elif name == 'vnews_content_nondupbd':
        df = df[['NEWS_ID', 'NEWS_ORIGIN_SOURCE', 'NEWS_TITLE', 'EFFECTIVE_TIME']]
        df.fillna({'NEWS_ORIGIN_SOURCE': '参考消息'}, inplace=True)
        # print(df)
    elif name == 'news_security_score':
        df = df[['NEWS_ID', 'SECURITY_INT_ID', 'SEC_SHORT_NAME', 'ASSET_CLASS', 'TICKER_SYMBOL', 'RELATED_DEGREE',
                 'RELATED_SCORE', 'DEGREE_PROP_1ST', 'DEGREE_PROP_2ED']]
    elif name == 'news_tag_v1':
        df = df[['NEWS_ID', 'NEWS_GENRE', 'IS_PRO_SITE', 'IS_MONTH_DATA', 'IS_POLICY', 'IS_PICTURE',
                 'IS_PERIOD', 'IS_WECHARTSTOCK', 'MINISTRY', 'INDUSTRY_NAME_1ST', 'INDUSTRY_NAME_2ND']]
    elif name == 'vnews_summary_v1':
        df = df[['NEWS_ID', 'NEWS_SUMMARY']]
        df = df[~df["NEWS_SUMMARY"].isna()]
        # df['NEWS_SUMMARY'] = df['NEWS_SUMMARY'].apply(lambda x: x.decode('utf-8'))

    df.to_parquet('after_data/' + name + '.parquet')


# 将快讯，微信，普通新闻整合
def merge_data():
    df1 = read_parquet('./after_data/vnews_content_nondupbd.parquet')
    df2 = read_parquet('./after_data/vnews_nondupbd_flash.parquet')
    df3 = read_parquet('./after_data/vnews_nondupbd_wechat.parquet')
    print(len(df1) + len(df2) + len(df3))
    df = pd.concat([df1, df2, df3])
    df['NEWS_ORIGIN_SOURCE'].fillna('未知', inplace=True)
    # print(df.head())
    df.to_parquet('after_data/vnews_nondupbd.parquet')


# 新闻数据过滤
def filter_data():
    df = read_parquet('./after_data/vnews_summary_v1.parquet')
    #
    # # 存在新闻标签过滤
    df1 = read_parquet('./after_data/vnews_content_nondupbd.parquet')
    df2 = read_parquet('./after_data/vnews_nondupbd_flash.parquet')
    df3 = read_parquet('./after_data/vnews_nondupbd_wechat.parquet')
    res1 = [x for x in df1['NEWS_ID'].apply(lambda x: x)]
    res2 = [x for x in df2['NEWS_ID'].apply(lambda x: x)]
    res3 = [x for x in df3['NEWS_ID'].apply(lambda x: x)]
    res1.extend(res2)
    res1.extend(res3)

    # 存在股票标签过滤
    stock = pd.read_csv('./market/stock.csv', encoding='gbk')
    id2stock = read_parquet('./after_data/news_security_score.parquet')
    stock_lists = [x for x in stock['股票代码'].apply(lambda x: x)]
    id2stock = id2stock[id2stock['SECURITY_INT_ID'].isin(stock_lists) == True]
    res4 = [x for x in id2stock['NEWS_ID'].apply(lambda x: x)]
    # print(res4)
    res = set(res1) & set(res4)

    df = df[df.NEWS_ID.isin(res) == True]

    df.to_parquet('after_data/vnews_summary.parquet')


# 生成存在标签的股票数据（3040类）
def comple_stock():
    df = pd.read_csv('./market/deal.csv', encoding='utf-8')[:3038]
    print(df[-1:])
    stock_lists = [x for x in df['order_book_id'].apply(lambda x: x)]
    sig = pd.read_csv('./market/SIGKDD.csv', encoding='gbk')
    sig = sig[sig['股票代码'].isin(stock_lists) == True]
    sig.to_csv('./market/sig.csv', index=False, encoding='gbk')


# 过滤股票信息中不存在标签的股票（2038类）
def filter_stock():
    df = pd.read_csv('./market/sig.csv', encoding='gbk')
    stock = read_parquet('./after_data/md_security.parquet')
    stock_lists = [x for x in stock['SEC_SHORT_NAME'].apply(lambda x: x)]

    df = df[df['股票名'].isin(stock_lists) == True]
    df.to_csv('./market/stock.csv', index=False, encoding='gbk')


# 过滤新闻股票信息中不存在标签的股票
# def filter_new2stock():
#     df = pd.read_csv('./market/sig.csv', encoding='gbk')
#     stock = read_parquet('./after_data/md_security.parquet')
#     stock_lists = [x for x in stock['SEC_SHORT_NAME'].apply(lambda x: x)]
#
#     df = df[df['股票名'].isin(stock_lists) == True]
#     df.to_csv('./market/stock.csv', index=False, encoding='gbk')


"""
这部分生成新闻预测标签的数据集
"""


# 组合数据用来预测标签
def combine_data1():
    df = read_parquet('./after_data/vnews_summary.parquet')
    news_list = set([x for x in df['NEWS_ID'].apply(lambda x: x)])

    # 需要删除无效新闻信息 √
    stock2id = read_parquet('./after_data/news_security_score.parquet')

    stock = pd.read_csv('./market/stock.csv', encoding='gbk')
    stock_lists = set([x for x in stock['股票代码'].apply(lambda x: x)])
    res = defaultdict(list)
    for index, row in stock2id.iterrows():
        a, b = row['NEWS_ID'], row['SECURITY_INT_ID']

        if a in news_list and b in stock_lists:
            res[a].append(b)
    print('stock list is finished')

    def fun(x):
        s = str(x[0])
        for i in range(1, len(x)):
            s += ',' + str(x[i])
        return s

    # df['label'] = df['NEWS_ID'].apply(lambda x: res[x])
    def cnt(y):
        return len(y)

    df['len'] = df['NEWS_ID'].apply(lambda x: cnt(res[x]))
    df = df[df['len'] <= 5]
    df['label'] = df['NEWS_ID'].apply(lambda x: fun(res[x]))
    df = df[['NEWS_ID', 'NEWS_SUMMARY', 'label']]
    df.to_parquet('filter/vnews_tag.parquet')


# 统计新闻预测标签数据，将其中涉及的股票生成onthot对应表
def make_onehot():
    df = pd.read_parquet('./filter/vnews_tag.parquet')
    stock_list = set()
    for index, row in df.iterrows():
        label = row['label'].split(',')
        label = set(label)
        stock_list = set.union(stock_list, label)

    res = {}
    num = 0
    for s in stock_list:
        res[s] = num
        num += 1

    dict_json = json.dumps(res, ensure_ascii=False)

    # 将json文件保存为.json格式文件

    with open('./market/id2stock1.json', 'w+', encoding='utf-8') as file:
        file.write(dict_json)


# 将股票标签转为one hot 编码用于预测
def turn_onehot():
    df = read_parquet('./filter/vnews_tag.parquet')

    with open('./market/id2onehot.json', 'r+', encoding='utf-8') as file:
        content = file.read()
    content = json.loads(content)

    def fun(s):
        # nonlocal content
        tmp = [x for x in s.split(',')]
        res = ['0'] * 2810
        for t in tmp:
            res[content[t]] = '1'
        res = ''.join(res)
        return res

    df.rename(columns={'label': 'stock'}, inplace=True)
    df['label'] = df['stock'].apply(lambda x: fun(x))
    df.reset_index(drop=True)
    df.to_parquet('filter/vnews_tag.parquet')


# 给融合新闻内容表，加入标题，时间，机构
def add_title():
    t_a = time.time()
    df = read_parquet('./filter/vnews_tag.parquet')
    print(len(df))
    addition = read_parquet('./after_data/vnews_nondupbd.parquet')
    tmp = pd.merge(left=df, right=addition, on="NEWS_ID")
    tmp['NEWS_SUMMARY'] = tmp['NEWS_TITLE'] + tmp['NEWS_SUMMARY']
    tmp = tmp[['NEWS_ID', 'NEWS_SUMMARY', 'label', 'stock', 'NEWS_ORIGIN_SOURCE', 'EFFECTIVE_TIME']]
    tmp.to_parquet('filter/vnews_tag1.parquet')
    print(len(tmp))
    t_b = time.time()
    print('cost time {}'.format(t_b - t_a))
    # def cnt(id, text):
    #     add = addition[(addition['NEWS_ID'] == int(id))]
    #     # print(len(add))
    #     if len(add) != 0:
    #         return text
    #     else:
    #         id, publish, title, date = add.values[:1][0]
    #         text = title + text
    #     return text
    #
    # df['NEWS_SUMMARY'] = df.apply(lambda row: cnt(row['NEWS_ID'], row['NEWS_SUMMARY']), axis=1)
    #
    # df.to_parquet('filter/vnews_tag1.parquet')


"""
这部分生成股票预测的数据集
"""


# 处理deal前期数据，将其股票编号转为内部股票编号，同时删除数据中不存在的股票信息
def filter_deal():
    df = pd.read_csv('./market/deal.csv')
    df = df[['date', 'order_book_id', 'pct1']]
    df = df[(df['date'] >= '2019') & (df['date'] <= '2023')]

    with open('./market/id2id.json', 'r+', encoding='utf-8') as file:
        content = file.read()
    content = json.loads(content)

    stock_lists = set([k for k, v in content.items()])
    # 删除参考数据中不存在的股票
    df = df[df['order_book_id'].isin(stock_lists) == True]
    # 股票代码替换
    df['order_book_id'] = df['order_book_id'].apply(lambda x: content[x])
    # 删除参考数据中不存在的日期
    ref = pd.read_csv('./market/reference.csv')
    ref = ref[(ref['date'] >= '2019') & (ref['date'] <= '2023')]
    date_list = set([x for x in ref['date'].apply(lambda x: str(x))])
    df = df[df['date'].isin(date_list) == True]

    df.to_csv('./market/deal.csv', index=False, encoding='utf-8')


# 处理deal数据，通过reference基准,生成每天股票的基准标签
def compile_deal1():
    df = pd.read_csv('./market/deal.csv')

    ref = pd.read_csv('./market/reference.csv')
    ref = ref[(ref['date'] >= '2019') & (ref['date'] <= '2023')]

    ref_data = {}
    for index, row in ref.iterrows():
        date, pct = row['date'], row['pct_1']
        ref_data[date] = pct

    df['ref_pct'] = df['date'].apply(lambda x: ref_data[str(x)])
    df['ref_pct'] = df['pct1'] - df['ref_pct']

    df.sort_values('ref_pct', ascending=False)
    q10 = df.quantile(.1, numeric_only=True).ref_pct
    q90 = df.quantile(.9, numeric_only=True).ref_pct

    def judge(x):
        if x <= q10:
            return 0
        elif x <= q90:
            return 1
        else:
            return 2

    df['label'] = df['ref_pct'].apply(lambda x: judge(x))
    df = df[['date', 'order_book_id', 'pct1', 'ref_pct', 'label']]
    df.to_csv('./market/deal.csv', index=False, encoding='utf-8')
    # res=sorted(set([x for x in df['date'].apply(lambda x:x)]))
    # print(res)


# 根据基准数据，对比相对于前一天的数据变化，生成真正的数据标签
def compile_deal2():
    with open('./market/id2id.json', 'r+', encoding='utf-8') as file:
        content = file.read()
    content = json.loads(content)

    stock = [v for k, v in content.items()]
    print(len(stock))
    df = pd.read_csv('./market/deal.csv')

    res = {}
    for s in stock:
        tmp = df.copy()
        tmp = tmp[tmp['order_book_id'] == s]

        # 根据排名变化1
        # cnt = [x for x in tmp['label'].apply(lambda x: x)]
        # 根据收益率变化
        cnt = [x for x in tmp['ref_pct'].apply(lambda x: x)]
        date = [x for x in tmp['date'].apply(lambda x: x)]

        tar = date[0] + '-' + str(s)
        res[tar] = 1
        tmp = cnt[0]
        for i in range(1, len(cnt)):
            tar = date[i] + '-' + str(s)
            # 排名
            # if tmp == cnt[i]:
            #     res[tar] = 1
            # elif tmp < cnt[i]:
            #     res[tar] = 2
            # else:
            #     res[tar] = 0

            # 收益率
            if tmp < cnt[i]:
                res[tar] = 1
            else:
                res[tar] = 0
            tmp = cnt[i]
    print('stock list is finish')

    def judge(day, id):
        tmp = day + '-' + str(id)
        return res[tmp]


#
#     df['label'] = df.apply(lambda x: judge(x.date, x.order_book_id), axis=1)
#     df = df[['date', 'order_book_id', 'pct1', 'label']]
#     df.to_csv('./market/deal.csv', index=False, encoding='utf-8')


# 将每条数据根据股票id拆分
def split_data():
    df = read_parquet('./filter/vnews_tag.parquet')
    df['EFFECTIVE_TIME'] = df['EFFECTIVE_TIME'].apply(lambda x: str(x).split(' ')[0])
    df = df[['NEWS_ID', 'NEWS_SUMMARY', 'stock', 'NEWS_ORIGIN_SOURCE', 'EFFECTIVE_TIME']]
    df.rename(columns={'EFFECTIVE_TIME': 'date', 'NEWS_ORIGIN_SOURCE': 'publish', 'NEWS_SUMMARY': 'content'}, inplace=1)
    df['len'] = df['stock'].apply(lambda x: len(x))
    for pos in range(1, 6):
        # pos = 1
        print(pos)
        tmp = df.copy()
        tmp = tmp[tmp['len'] >= pos]
        tmp['stock_id'] = tmp['stock'].apply(lambda x: x[pos - 1])
        tmp = tmp[['content', 'publish', 'date', 'stock_id']]
        tmp.to_parquet('filter/vnews_stock{}.parquet'.format(pos))


def concat_stock():
    df1 = read_parquet('./filter/vnews_stock1.parquet')
    df2 = read_parquet('./filter/vnews_stock2.parquet')
    df3 = read_parquet('./filter/vnews_stock3.parquet')
    df4 = read_parquet('./filter/vnews_stock4.parquet')
    df5 = read_parquet('./filter/vnews_stock5.parquet')
    print(len(df1))
    print(len(df2))
    print(len(df3))
    print(len(df4))
    print(len(df5))
    df = pd.concat([df1, df2, df3, df4, df5])
    df.to_parquet('filter/vnews_stock.parquet')


# 组合数据用来股票预测
def combine_data2():
    df = read_parquet('./filter/vnews_stock.parquet')

    ref = pd.read_csv('./market/deal.csv')
    ref = ref[['date', 'order_book_id', 'label']]
    ref.rename(columns={'order_book_id': 'stock_id'}, inplace=1)
    df = pd.merge(df, ref)
    df = df[['date', 'publish', 'content', 'stock_id', 'label']]
    df = df.drop_duplicates(subset=['content'])
    df = df.reset_index(drop=True)

    # 生成新闻发布机构的onthot编码,初次使用生成
    res = set([x for x in df['publish'].apply(lambda x: x)])
    res = set(res)
    content = {r: index for index, r in enumerate(res)}
    dict_json = json.dumps(content, ensure_ascii=False)
    with open('./market/publish2onthot.json', 'w+', encoding='utf-8') as file:
        file.write(dict_json)

    # 将新闻数据替换成onthot编码
    with open('./market/publish2onthot.json', 'r+', encoding='utf-8') as file:
        content = file.read()
    content = json.loads(content)
    df['publish'] = df['publish'].apply(lambda x: content[x])

    # 将同一天的股票新闻与机构合并
    df1 = df.groupby(['date', 'stock_id'])['content'].apply(lambda grp: list(grp)).reset_index()
    df2 = df.groupby(['date', 'stock_id'])['publish'].apply(lambda grp: list(grp)).reset_index()
    df3 = df.groupby(['date', 'stock_id'])['label'].apply(lambda grp: list(grp)[0]).reset_index()
    df = pd.merge(df1, df2)
    df = pd.merge(df, df3)
    df['len'] = df['publish'].apply(lambda x: len(x))
    df.to_parquet('filter/vnews_stock.parquet')


if __name__ == '__main__':
    file_names = {0: 'md_security.parquet', 1: 'news_security_score.parquet', 2: 'news_tag_v1.parquet',
                  3: 'vnews_content_nondupbd.parquet', 4: 'vnews_nondupbd_flash.parquet',
                  5: 'vnews_nondupbd_wechat.parquet',
                  6: 'vnews_summary_v1.parquet'}

    # 数据初步筛选
    # file_pos = 3
    # merge_data()
    # parquet_csv('before_data/' + file_names[file_pos])
    # print('{} is finished'.format(file_names[file_pos]))
    # look_data(file_names[file_pos], 2)

    # filter_data()
    # combine_data1()
    # combine_data1_add()
    # turn_onehot()
    # add_title()

    # filter_deal()
    # compile_deal1()
    # split_data()
    # concat_stock()
    combine_data2()
