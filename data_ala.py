import json

import pandas as pd

import os
from collections import defaultdict, Counter
from pandas import read_parquet, read_csv

os.environ['CUDA_VISIBLE_DEVICES'] = '6'


def fun():
    df = read_parquet('./filter/vnews_stock.parquet')
    df['len'] = df['content'].apply(lambda x: len(str(x)))
    print(df['len'].describe([.5, .7, .9]))
    # print(len(df2))
    # print(df.head())
    # print(df2.head())
    # def get_stock(a):
    #     a = list(a)
    #     r = []
    #     for index, t in enumerate(a):
    #         if t == '1':
    #             r.append(index)
    #     return r
    #
    # df['stock'] = df['label'].apply(lambda x: get_stock(x))
    # df.reset_index(drop=True)
    # df.to_parquet('filter/vnews_tag1.parquet')


def look_data():
    df = pd.read_parquet('./filter/vnews_stock.parquet')

    print(df.head())
    print(len(df))
    # df = read_parquet('./after_data/' + path)
    # df=df[['content','label']]


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


def news_withtag():
    tag = read_parquet('./filter/vnews_stock.parquet')

    res = [x for x in tag['SECURITY_INT_ID'].apply(lambda x: x)]
    res = set(res)
    print(len(res))


def tag2list():
    tag = read_parquet('./after_data/md_security.parquet')

    res1, res2 = {}, {}

    for index, row in tag.iterrows():
        res1[int(row['SECURITY_ID'])] = row['SEC_SHORT_NAME']
        res2[row['SEC_SHORT_NAME']] = row['SECURITY_ID']

    dict_json1 = json.dumps(res1, ensure_ascii=False)
    dict_json2 = json.dumps(res2, ensure_ascii=False)

    # 将json文件保存为.json格式文件

    with open('./market/id2stock.json', 'w+', encoding='utf-8') as file:
        file.write(dict_json1)


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

    file_pos = 3
    # stockid()

    # 数据观察
    # look_data(file_names[file_pos])
    # look_data('vnews_tag.parquet')
    look_data()
    # new_deal()

    # 时间跨度观察
    # news_withdata()

    # 新闻机构观察
    # news_withpublish()

    # 新闻对应标签
    # news_withtag()
    # with open('./after_data/stock.json', 'r+', encoding='utf-8') as file:
    #     content = file.read()
    #
    # content = json.loads(content)
