import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
from collections import defaultdict, Counter

from pandas import read_parquet

os.environ['CUDA_VISIBLE_DEVICES'] = '6'


def look_data(path):
    df = read_parquet('./filter/' + path)
    # df['len'] = df['NEWS_SUMMARY'].apply(lambda x: len(x))
    print(len(df))
    # print(df.columns)
    print(df.head())
    # df = df[['NEWS_ORIGIN_SOURCE', 'NEWS_TITLE']]
    # for index, row in df.iterrows():
    #     a, b = row['NEWS_ORIGIN_SOURCE'], row['NEWS_TITLE']
    #     print(a, b)
    # if a != b:
    #     print(a, b)
    # for index, row in df.iterrows():
    #     b = row['SEC_SHORT_NAME']
    #     # print(b)
    #     if '万科A' == b:
    #         print(1)



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


def news_withtag():
    tag = read_parquet('./after_data/news_security_score.parquet')

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
    with open('./market/stock2id.json', 'w+', encoding='utf-8') as file:
        file.write(dict_json2)


if __name__ == '__main__':
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    file_names = {0: 'md_security.parquet', 1: 'news_security_score.parquet', 2: 'news_tag_v1.parquet',
                  3: 'vnews_content_nondupbd.parquet', 4: 'vnews_nondupbd_flash.parquet',
                  5: 'vnews_nondupbd_wechat.parquet',
                  6: 'vnews_summary_v1.parquet'}

    file_pos = 0
    # 数据观察
    # look_data('vnews_tag.parquet')
    look_data('vnews_tag_1.parquet')

    # 时间跨度观察
    # news_withdata()

    # 新闻对应标签
    # news_withtag()
    # with open('./after_data/stock.json', 'r+', encoding='utf-8') as file:
    #     content = file.read()
    #
    # content = json.loads(content)
    #
    # tags = {68617281: [10004120, 2112, 3334],
    #         68617312: [29366, 3511, 2141852, 20005199],
    #         68617323: [4608, 2945, 30785, 3715, 75524, 34886, 3783, 2439, 70022, 1032, 802821, 30012864, 73032, 70667,
    #                    39946, 10002963, 30011846, 273, 39891, 71252, 86, 2199, 75158, 39705, 3420, 71518, 30010793,
    #                    39009, 35362, 4130, 2980, 39973, 1704, 20005083, 72365, 3437, 749, 30012836, 5551, 2288, 29107,
    #                    30581, 75321, 31803],
    #         68617322: [3489, 2945, 35362, 38664, 10002964, 749, 5551, 273, 86, 2199, 75158, 2363, 3420],
    #         68617388: [1986, 203205, 200]}
    # pos = [68617281, 68617312, 68617323, 68617322, 68617388]
    #
    # df = read_parquet('./after_data/' + file_names[file_pos])
    # df = df[df.NEWS_ID.isin(pos) == True]
    # for index, row in df.iterrows():
    #     print(row['NEWS_SUMMARY'])

    # tag2list()

'''
    前期数据
    md_security                       股票机构信息    1416998类   177400类(有效)  19499类(频繁类)
    news_security_score.parquet       新闻股票对应   82203860    28523727(去重)
    news_tag_v1.parquet               新闻标签(暂不用)
    vnews_content_nondupbd.parquet    普通新闻
    vnews_nondupbd_flash.parquet      快讯新闻
    vnews_nondupbd_wechat.parquet     微信新闻      Counter({'普通微信号': 1917358, '上市公司微信号': 568848, '非宏观券商微信号': 56243, None: 205})
    vnews_summary_v1.parquet          新闻主体    15G  1亿
    vnews_summary.parquet             新闻主体    4G  0.27亿
    sig.csv                           预测股票标签 2900+
    
    模型预测
    vnews_tag.parquet                 新闻与预测标签（步骤1：分类）  7070721
'''
