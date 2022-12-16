import json

import pandas as pd
import os

from pandas import read_parquet
from collections import defaultdict

os.environ['CUDA_VISIBLE_DEVICES'] = '6'


# 不同文件编码转化
def compile_id():
    df = pd.read_csv('./market/stock1.csv', encoding='gbk')

    with open('./market/stock2id.json', 'r+', encoding='utf-8') as file:
        content = file.read()
    content = json.loads(content)

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

    with open('./market/stock2id.json', 'r+', encoding='utf-8') as file:
        content = file.read()
    content = json.loads(content)

    df['股票代码'] = df['股票名'].apply(lambda x: content[x])
    df.to_csv('./market/stock.csv', index=False, encoding='gbk')


# 转换市场对应编码
def compile_deal():
    df = pd.read_csv('./market/deal.csv', encoding='utf-8')
    df = df[['date', 'order_book_id', 'pct1', 'pct3', 'pct5', 'pct10']]

    with open('./market/id2id.json', 'r+', encoding='utf-8') as file:
        content = file.read()
    content = json.loads(content)
    stock_lists = [k for k, v in content.items()]

    df = df[df['order_book_id'].isin(stock_lists) == True]
    df['order_book_id'] = df['order_book_id'].apply(lambda x: content[x])
    df.to_csv('./market/deal.csv', index=False)


def parquet_csv(path):
    df = read_parquet(path)
    filename = path.split('/')[-1]
    name = filename.split('.')[0]
    if name == 'md_security':
        df = df[['SECURITY_ID', 'SEC_FULL_NAME', 'SEC_SHORT_NAME', 'LIST_STATUS_CD']]
    elif name == 'vnews_nondupbd_flash':
        df = df[['NEWS_ID', 'NEWS_ORIGIN_SOURCE', 'NEWS_TITLE', 'NEWS_PUBLISH_SITE', 'EFFECTIVE_TIME']]
        df = df[['NEWS_ID', 'NEWS_TITLE', 'NEWS_PUBLISH_SITE', 'EFFECTIVE_TIME']]
        # df.columns = ['NEWS_ID', 'NEWS_TITLE', 'NEWS_ORIGIN_SOURCE', 'EFFECTIVE_TIME']
        df.fillna({'NEWS_ORIGIN_SOURCE': '参考消息'}, inplace=True)
    elif name == 'vnews_nondupbd_wechat':
        df = df[['NEWS_ID', 'NEWS_ORIGIN_SOURCE', 'NEWS_TITLE', 'SITE_TYPE', 'EFFECTIVE_TIME']]
        df.fillna({'NEWS_ORIGIN_SOURCE': '参考消息'}, inplace=True)
    elif name == 'vnews_content_nondupbd':
        df = df[['NEWS_ID', 'NEWS_ORIGIN_SOURCE', 'NEWS_TITLE', 'EFFECTIVE_TIME']][:200]
        df.fillna({'NEWS_ORIGIN_SOURCE': '参考消息'}, inplace=True)
        print(df)
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


def look_data(path, mode=1):
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    if mode == 1:
        df = read_parquet('./before_data/' + path)[:10]
    else:
        df = read_parquet('./after_data/' + path)[:10]
    # print(df.columns.values)
    print(df.head())


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


# 组合数据用来预测标签（暂未加入标题数据）
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

    # def fun(x):
    #     s = str(x[0])
    #     for i in range(1, len(x)):
    #         s += ',' + str(x[i])
    #     return s

    df['label'] = df['NEWS_ID'].apply(lambda x: res[x])
    df.to_parquet('filter/vnews_tag.parquet')


if __name__ == '__main__':
    file_names = {0: 'md_security.parquet', 1: 'news_security_score.parquet', 2: 'news_tag_v1.parquet',
                  3: 'vnews_content_nondupbd.parquet', 4: 'vnews_nondupbd_flash.parquet',
                  5: 'vnews_nondupbd_wechat.parquet',
                  6: 'vnews_summary_v1.parquet'}

    # 数据初步筛选
    # file_pos = 3
    # look_data(file_names[file_pos])
    # parquet_csv('before_data/' + file_names[file_pos])
    # print('{} is finished'.format(file_names[file_pos]))
    # look_data(file_names[file_pos], 2)

    # filter_data()
    combine_data1()
