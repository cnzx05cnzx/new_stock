import argparse
import datetime
import os
import pickle
import time
import pandas as pd
import random
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics, preprocessing
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import math
from collections import defaultdict

from util_hanlp import *

os.environ['CUDA_VISIBLE_DEVICES'] = '5'


def seed_torch(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = True


def XGB_model(path, data_x, data_id, y):
    t_a = time.time()
    vis = dict(zip(y['order_book_id'], y['pct']))
    id = data_id
    x = data_x
    # frquent  ref_pct
    t_b = time.time()
    print('data load cost {:.2f} s'.format(t_b - t_a))
    x = np.array(x)

    with open(path, 'rb') as f:
        loaded_model = pickle.load(f)

    res = loaded_model.predict_proba(x)

    up = [math.log(x / (1 - x)) for x in res[:, 2]]
    down = [math.log(x / (1 - x)) for x in res[:, 0]]

    res = [u - p for u, p in zip(up, down)]
    res = [(r, i) for r, i in zip(res, id)]

    d = defaultdict(list)
    for k, v in res:
        if v in vis:
            d[v].append(k)

    res = [(sum(v) / len(v), vis[k]) for k, v in d.items()]

    # 按照测定值排序  (a,b)  a计算的值  b实际的值
    sorted_list = sorted(res, key=lambda x: -x[0])
    # 按照真实值排序
    # sorted_list = sorted(res, key=lambda x: -x[1])

    # 分成5类
    parts = []
    pos = 0
    n = len(sorted_list)
    print('all stock have {} type'.format(n))

    # 分成 5类 2，11，2  分成10类  1，11，1
    for i in range(2, 11, 2):
        tmp = int(i * n * 0.1)
        parts.append(sorted_list[pos:tmp])
        pos = tmp

    part_means = [sum([p[1] for p in part]) / len(part) for part in parts]
    # part_means = normalize_list(part_means)
    print(part_means)

    return part_means


def draw_layer(data, date, gap):
    print('final res is {:.2f}'.format((data[1][-1] - data[5][-1]) / len(data[1])))
    for index, (k, v) in enumerate(data.items()):
        # if index == 0 or index == 4:
        plt.plot(date, v, label=k)

    # 添加标签和标题
    plt.xlabel('time')
    plt.ylabel('real ref')
    # plt.title('折线图')
    ticks = [i for i in range(1, len(date), gap)]
    use_date = []
    for index, d in enumerate(date, start=1):
        if index in ticks:
            use_date.append(d)
    plt.xticks(ticks, use_date, rotation=15, fontsize=10)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--EPOCHS', type=int, default=3)
    parser.add_argument('--BATCH_SIZE', type=int, default=64)
    parser.add_argument('--LEARNING_RATE', type=float, default=1e-5)
    parser.add_argument('--MAX_LEN', default=140)
    parser.add_argument('--GAP', default=5)
    parser.add_argument('--PATH', default='./model/vnews_stock_xgboost.pickle')

    args = parser.parse_args()
    seed_torch(args.seed)

    df = pd.read_parquet('./filter/vnews_stock_split.parquet')
    deal = pd.read_csv('./market/deal.csv')
    df['date'] = pd.to_datetime(df['date']).dt.date

    layer = defaultdict(list)
    date_list = []
    start_date = datetime.date(2022, 5, 1)
    end_date = datetime.date(2022, 6, 30)
    # 真的需要重新进行词频提取嘛，不会损失模型的性能？
    current_date = start_date

    while current_date <= end_date:
        now_day = current_date.strftime("%Y-%m-%d")
        print(now_day)
        train_day = current_date - datetime.timedelta(days=30)

        tmp_deal = deal.copy()
        tmp_deal = tmp_deal[tmp_deal['date'].str.contains(now_day)]
        tmp_deal = tmp_deal[['order_book_id', 'pct']]
        tmp_deal.reset_index(drop=True, inplace=True)
        if len(tmp_deal) > 1:

            tmp = df.copy()
            mask = (tmp['date'] >= train_day) & (tmp['date'] <= current_date)
            tmp_data = tmp.loc[mask]
            tmp_data.to_parquet('filter/vnews_stock_tmp.parquet')
            # 获取高频词表
            hanlp2word(tmp_data)
            del tmp_data
            # 计算词频特征
            d_x, d_id = get_vector('filter/vnews_stock_tmp.parquet', './word_split/output_hanlp.txt', 'test')

            date_list.append(now_day)

            model_path = './model/vnews_stock_xgboost_{}.pickle'.format(now_day)
            # model_path=args.PATH
            layer_res = XGB_model(model_path, d_x, d_id, tmp_deal)

            for index, lay in enumerate(layer_res, start=1):
                layer[index].append(lay)

        current_date += datetime.timedelta(days=1)

    # 累加收益

    m, n = len(layer), len(layer[1])
    # print(m,n)
    for i in range(1, m + 1):
        for j in range(1, n):
            # print(layer[i][j])
            layer[i][j] += layer[i][j - 1]
            # print(layer[i][j])

    draw_layer(layer, date_list, args.GAP)
