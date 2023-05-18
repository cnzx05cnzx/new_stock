import json

import pandas as pd
import torch
import os
from collections import defaultdict, Counter
from pandas import read_parquet, read_csv

os.environ['CUDA_VISIBLE_DEVICES'] = '6'


def fun():
    df = read_parquet('./after_data/news_tag_v1.parquet')

    # df = df[df['IS_POLICY'] == True]
    ref = read_parquet('./after_data/vnews_summary.parquet')

    tp = pd.merge(df, ref)
    print(tp.head())
    tp = tp[tp['NEWS_GENRE'] == '普通新闻']
    for index, row in tp[:100].iterrows():
        print(row['NEWS_SUMMARY'])


def look_data():
    df = read_parquet('./filter/vnews_stock_merge2.parquet')
    # df = read_parquet('./market/index000905_pct2_t-1close_buy_t+1closesell.parquet')
    # df = read_parquet('./market/pct2_t-1close_buy_t+1closesell.parquet')
    # df = pd.read_csv('./market/dataset_day.csv')
    print(len(df))
    print(df.head())
    # res=df.groupby(['date', 'stock_id'])
    # print(len(res))
    # df = read_parquet('./filter/vnews_stock_split.parquet')
    # df = df[df['date'].str.contains('2022-09')]
    # print(df.head())
    for index, item in df[:100].iterrows():
        print(item['date'])


def news_withdata():
    df1 = read_parquet('./after_data/vnews_content_nondupbd.parquet')
    df2 = read_parquet('./after_data/vnews_nondupbd_flash.parquet')
    df3 = read_parquet('./after_data/vnews_nondupbd_wechat.parquet')
    cnt = len(df1) + len(df2) + len(df3)
    print('all tagged news have {}'.format(cnt))
    date1 = sorted([x for x in df1['EFFECTIVE_TIME'].apply(lambda x: x)])
    date2 = sorted([x for x in df2['EFFECTIVE_TIME'].apply(lambda x: x)])
    date3 = sorted([x for x in df3['EFFECTIVE_TIME'].apply(lambda x: x)])
    print(date1[0], date1[-1])
    print(date2[0], date2[-1])
    print(date3[0], date3[-1])


def news_withpublish():
    df1 = read_parquet('./after_data/vnews_content_nondupbd.parquet')
    df2 = read_parquet('./after_data/vnews_nondupbd_flash.parquet')
    df3 = read_parquet('./after_data/vnews_nondupbd_wechat.parquet')
    print(df1.columns)
    print(df2.columns)
    print(df3.columns)

    date1 = set([x for x in df1['NEWS_ORIGIN_SOURCE'].apply(lambda x: x)])
    date2 = set([x for x in df2['NEWS_PUBLISH_SITE'].apply(lambda x: x)])
    date3 = set([x for x in df3['NEWS_ORIGIN_SOURCE'].apply(lambda x: x)])

    data = date1.union(date2)
    data = data.union(date3)
    print(len(data))


def hot_compute():
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [1, 2, 4]})
    print(df)


# 查看deal市场股票id与打标数据是否一致
def stockid():
    with open('./market/id2onehot.json', 'r+', encoding='utf-8') as file:
        content = file.read()
    content = json.loads(content)
    stock_lists = [k for k, v in content.items()]
    df = pd.read_csv('./market/deal.csv')

    df = df[df['date'] == '2017-01-03']
    res = [x for x in df['order_book_id'].apply(lambda x: x)]
    res = set(res)
    print(len(res))
    num = 0
    # for s in stock_lists:
    #     if int(s) not in res:
    #         # print(s)
    #         num+=1
    for s in res:
        if str(s) not in content:
            # print(s)
            num += 1
    print(num)
    # print(res)
    # print(df.head())


def new_deal():
    df = pd.read_csv('./market/deal.csv')
    df = df[df['order_book_id'] == 2]
    df = df[df['date'].str.contains('2019')]
    res = [x for x in df['label'].apply(lambda x: x)]
    print(res)

    # print(cnt)


if __name__ == '__main__':
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    file_names = {0: 'md_security.parquet', 1: 'news_security_score.parquet', 2: 'news_tag_v1.parquet',
                  3: 'vnews_content_nondupbd.parquet', 4: 'vnews_nondupbd_flash.parquet',
                  5: 'vnews_nondupbd_wechat.parquet',
                  6: 'vnews_summary_v1.parquet'}

    # stockid()

    # 数据观察
    # look_data('vnews_tag.parquet')
    look_data()
    # fun()

    # 时间跨度观察
    # news_withdata()

    # hot_compute()
    # print(torch.cuda.is_available())
