import datetime
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import math
import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import random
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '6'


def seed_torch(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


class StockDataset_many(Dataset):

    def __init__(self, dataframe, tokenizer, max_len, new_data=False):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.text = dataframe.content
        self.publish = dataframe.publish
        self.stock_id = dataframe.stock_id
        self.new_data = new_data
        self.ref_pct = dataframe.ref_pct
        self.max_len = max_len

        if not new_data:
            self.targets = self.data.label

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index][:6]
        publish = self.publish[index][:6]
        ref_pct = self.ref_pct[index]
        stock_id = self.stock_id[index]
        inputs_list, publish_list = [], []
        for t, p in zip(text, publish):
            inputs = self.tokenizer(
                t,
                truncation=True,
                padding='max_length' if self.new_data else False,
                max_length=self.max_len,
                return_tensors="pt"
            )
            inputs = {k: v.squeeze() for k, v in inputs.items()}
            inputs_list.append(inputs)
            publish_list.append(p)

        if not self.new_data:
            labels = self.targets[index]
            return inputs_list, publish_list, labels

        return inputs_list, publish_list, ref_pct, stock_id


class News_Stock(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_CKPT)
        # self.embedding = nn.Embedding(1000, 32)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 3)
        )

    def forward(self, inputs, publish, pos, mode='train', label=None):

        bert_output = self.bert(**inputs)
        hidden_state = bert_output.last_hidden_state
        pooled_out = hidden_state[:, 0]
        # key_bedding = self.embedding(publish)
        # all_out = torch.cat([pooled_out, key_bedding], dim=1)
        all_out = pooled_out
        logits = self.classifier(all_out)
        zero, one, two = torch.tensor([0]).to(device), torch.tensor([1]).to(device), torch.tensor([2]).to(device)
        if mode == 'train':
            output = torch.empty(0, 3).to(device)
            left = 0
            for p in pos:
                right = p.item()
                tmp = logits[left:right]
                # true = label[left:right]
                # tar = torch.max(tmp.data, 1)[1]
                tar1 = torch.kthvalue(tmp.data, 1)[1]
                tar2 = torch.kthvalue(tmp.data, 2)[1]

                tar = torch.cat((tar1, tar2), dim=0)
                a, b, c = torch.bincount(tar, minlength=3)
                if c >= a and c >= b:
                    # tar = tar.to(device)
                    vote_label = two
                elif a >= c and a >= b:
                    vote_label = zero
                else:
                    vote_label = one

                # a, b = torch.bincount(tar, minlength=2)
                # if a >= b:
                #     # tar = tar.to(device)
                #     vote_label = zero
                # else:
                #     vote_label = one

                vote_label = vote_label.repeat(right - left)
                # id = torch.where((tar == vote_label).type(torch.int))[0]
                vote1 = torch.where((tar1 == vote_label).type(torch.int))[0]
                vote2 = torch.where((tar2 == vote_label).type(torch.int))[0]
                id = torch.unique(torch.cat([vote1, vote2], dim=0))

                tmp = tmp[id]
                res = torch.mean(tmp, dim=0).unsqueeze(0)

                output = torch.cat((output, res), dim=0)
                left = right
            # predict
        elif mode == 'eval':
            output = torch.tensor([]).to(device)
            left = 0
            for p in pos:
                right = p.item()
                tmp = logits[left:right]
                # true = label[left:right]
                # tar = torch.max(tmp.data, 1)[1]
                tar = torch.topk(tmp.data, k=2, dim=1)[1].view(1, -1).squeeze(0)

                a, b, c = torch.bincount(tar, minlength=3)
                if c >= a and c >= b:
                    # tar = tar.to(device)
                    vote_label = two
                elif a >= c and a >= b:
                    vote_label = zero
                else:
                    vote_label = one
                # a, b = torch.bincount(tar, minlength=2)
                # if a >= b:
                #     # tar = tar.to(device)
                #     vote_label = zero
                # else:
                #     vote_label = one

                output = torch.cat((output, vote_label), dim=0)
                left = right
        else:
            output = logits
        return output


# def BERT_model(path, data):
#     t_a = time.time()
#     x = data['frquent'].values.tolist()  # 将列的元素直接转换为列表
#     y = data['ref_pct'].values.tolist()  # 将列的元素直接转换为列表
#     # frquent  ref_pct
#     t_b = time.time()
#     print('data load cost {:.2f} min'.format((t_b - t_a) / 60))
#     x = np.array(x)
#
#     n = len(x)
#     print(n)
#
#     with open(path, 'rb') as f:
#         loaded_model = pickle.load(f)
#
#     res = loaded_model.predict_proba(x)
#     # print(res)
#
#     up = [math.log(x / (1 - x)) for x in res[:, 2]]
#     down = [math.log(x / (1 - x)) for x in res[:, 0]]
#     # res = res[:, 2] - res[:, 0]
#     res = [u - p for u, p in zip(up, down)]
#     res = [(r, ref) for r, ref in zip(res, y)]
#
#     sorted_list = sorted(res, key=lambda x: -x[0])
#
#     # 分成5类
#     parts = []
#     pos = 0
#     for i in range(2, 11, 2):
#         tmp = int(i * n * 0.1)
#         parts.append(sorted_list[pos:tmp])
#         pos = tmp
#     # print(parts)
#     # 扩大100倍
#     part_means = [sum([p[1] for p in part]) / len(part) for part in parts]
#     # part_means = normalize_list(part_means)
#     # print(part_means)
#
#     return part_means


def BERT_model(model, test_iter_many, y):
    model.eval()
    vis = dict(zip(y['order_book_id'], y['pct']))
    d = defaultdict(list)

    with torch.no_grad():
        for i, (data, publish, pos, ref_pct, stock_id) in enumerate(test_iter_many):
            pos = pos.to(device)
            outputs = model(data.to(device), publish.to(device), pos, 'train')
            outputs = F.softmax(outputs, dim=1)
            outputs = outputs.detach().cpu().numpy()
            ref = ref_pct.numpy()
            stock_id = stock_id.numpy()
            up = [math.log(x / (1 - x)) for x in outputs[:, 2]]
            down = [math.log(x / (1 - x)) for x in outputs[:, 0]]

            tmp = [u - p for u, p in zip(up, down)]

            for k, v in zip(stock_id, tmp):
                if int(k) in vis:
                    d[int(k)].append(v)
            # tmp = [(r, ref) for r, ref in zip(tmp, ref)]
            # res.extend(tmp)
    res = [(sum(v) / len(v), vis[k]) for k, v in d.items()]
    sorted_list = sorted(res, key=lambda x: -x[0])
    n = len(sorted_list)
    # print(1)
    # 分成5类
    parts = []
    pos = 0
    for i in range(2, 11, 2):
        tmp = int(i * n * 0.1)
        parts.append(sorted_list[pos:tmp])
        pos = tmp
    # print(parts)
    # 扩大100倍
    part_means = [sum([p[1] for p in part]) / len(part) for part in parts]
    # part_means = normalize_list(part_means)
    return part_means


def judge(x, a, b):
    if x <= a:
        return 0
    elif x <= b:
        return 1
    else:
        return 2


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
    parser.add_argument('--PATH', default='./model/vnews_stock_1.pkl')
    parser.add_argument('--GAP', default=5)
    args = parser.parse_args()
    seed_torch(args.seed)

    MODEL_CKPT = 'hfl/chinese-bert-wwm-ext'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)

    df = pd.read_parquet('./filter/vnews_stock_merge.parquet')
    df['date'] = pd.to_datetime(df['date']).dt.date
    deal = pd.read_csv('./market/deal.csv')
    year = 2022


    def dynamic_collate_3(data):

        inputs_list, keyword_list, pos_list, ref_pct, stock_id = [], [], [0], [], []
        for inputs, k, ref, id in data:
            inputs_list.extend(inputs)
            keyword_list.extend(k)
            num = len(k)
            pos_list.append(num + pos_list[-1])
            ref_pct.append(ref)
            stock_id.append(id)

        inputs_list = tokenizer.pad(inputs_list, return_tensors='pt')

        keyword_list = torch.tensor(keyword_list, dtype=torch.long)
        pos_list = torch.tensor(pos_list, dtype=torch.long)
        ref_list = torch.tensor(ref_pct, dtype=torch.float)
        stock_id = torch.tensor(stock_id, dtype=torch.float)
        return inputs_list, keyword_list, pos_list[1:], ref_list, stock_id


    month = 1
    layer = {
        1: [], 2: [], 3: [], 4: [], 5: []
    }
    date_list = []
    start_date = datetime.date(2022, 5, 1)
    end_date = datetime.date(2022, 6, 30)
    current_date = start_date

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("device {}".format(device))

    model = News_Stock()
    model.load_state_dict(torch.load('./model/vnews_stock.pkl'))
    model.to(device)
    while current_date <= end_date:
        now_day = current_date.strftime("%Y-%m-%d")
        tmp_deal = deal.copy()
        mask = (tmp_deal['date'] == current_date)
        tmp_deal = tmp_deal.loc[mask]
        tmp_deal.reset_index(drop=True, inplace=True)
        if len(tmp_deal) > 5:
            tmp = df.copy()
            train_day = current_date - datetime.timedelta(days=30)
            mask = (tmp['date'] > train_day) & (tmp['date'] <= current_date)
            tmp_df = tmp.loc[mask]
            tmp_data = tmp_df[['stock_id', 'publish', 'content', 'ref_pct', 'len']].copy()
            del tmp_df

            date_list.append(now_day)
            print('-' * 10)
            print('data len {}'.format(len(tmp_data)))
            tmp_data.reset_index(drop=True, inplace=True)
            test_set = StockDataset_many(tmp_data, tokenizer, args.MAX_LEN, new_data=True)
            test_loader = DataLoader(test_set, batch_size=20, collate_fn=dynamic_collate_3)

            layer_res = BERT_model(model, test_loader, tmp_deal)
            print(current_date)
            print(layer_res)

            for index, lay in enumerate(layer_res, start=1):
                layer[index].append(lay)
        # break
        current_date += datetime.timedelta(days=1)

    m, n = len(layer), len(layer[1])
    # print(m,n)
    for i in range(1, m + 1):
        for j in range(1, n):
            layer[i][j] += layer[i][j - 1]

    draw_layer(layer, date_list, args.GAP)
