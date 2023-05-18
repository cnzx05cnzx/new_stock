import argparse
import collections
import re
import os
import string
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import random
import torch.nn.functional as F
from sklearn.metrics import f1_score

os.environ['CUDA_VISIBLE_DEVICES'] = '5'


def seed_torch(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


class StockDataset_one(Dataset):

    def __init__(self, dataframe, tokenizer, max_len, new_data=False):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.text = dataframe.content
        self.publish = dataframe.publish
        self.new_data = new_data
        self.max_len = max_len

        if not new_data:
            self.targets = self.data.label

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index][0]
        publish = self.publish[index][0]

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length' if self.new_data else False,
            max_length=self.max_len,
            return_tensors="pt"
        )
        inputs = {k: v.squeeze() for k, v in inputs.items()}

        if not self.new_data:
            labels = torch.tensor(self.targets[index], dtype=torch.long)
            return inputs, publish, labels

        return inputs, publish


class StockDataset_many(Dataset):

    def __init__(self, dataframe, tokenizer, max_len, new_data=False):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.text = dataframe.content
        self.publish = dataframe.publish
        self.new_data = new_data
        self.max_len = max_len

        if not new_data:
            self.targets = self.data.label

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index][:8]
        publish = self.publish[index][:8]
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

        return inputs_list, publish_list


class News_Stock(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_CKPT)
        self.embedding = nn.Embedding(28592, 32)
        self.classifier = nn.Sequential(
            nn.Linear(800, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 3)
        )
        self.output = torch.tensor([])

    def forward(self, inputs, publish, pos, mode='one', label=None):
        if mode == 'one':
            bert_output = self.bert(**inputs)
            hidden_state = bert_output.last_hidden_state
            pooled_out = hidden_state[:, 0]
            key_bedding = self.embedding(publish)
            all_out = torch.cat([pooled_out, key_bedding], dim=1)
            # all_out=pooled_out
            logits = self.classifier(all_out)
            return logits
        elif mode == 'many':
            bert_output = self.bert(**inputs)
            hidden_state = bert_output.last_hidden_state
            pooled_out = hidden_state[:, 0]
            key_bedding = self.embedding(publish)
            all_out = torch.cat([pooled_out, key_bedding], dim=1)
            # all_out = pooled_out
            logits = self.classifier(all_out)

            # train or eval
            if label is not None:
                left = 0

                output = torch.empty(0, 3).to(device)
                for p in pos:
                    right = p.item()
                    tmp = logits[left:right]
                    true = label[left:right]
                    tar = torch.max(tmp.data, 1)[1]
                    id = torch.where((tar == true).type(torch.int))

                    if id[0].numel():
                        tmp = tmp[id]
                    res = torch.mean(tmp, dim=0).unsqueeze(0)

                    output = torch.cat((output, res), dim=0)
                    left = right
            # predict
            else:
                output = logits
            return output


def train(model, train_iter_one, train_iter_many, dev_iter_one, dev_iter_many, args, opt):
    best_f1 = float(0)
    cnt = 0  # 记录多久没有模型效果提升
    stop_flag = False  # 早停标签

    for epoch in range(args.EPOCHS):
        print('Epoch [{}/{}]'.format(epoch + 1, args.EPOCHS))

        # 训练-------------------------------
        model.train()
        t_a = time.time()
        total_loss1, total_loss2 = 0, 0

        for i, (data, publish, targets) in enumerate(train_iter_one):
            outputs = model(data.to(device), publish.to(device), 'one')
            targets = targets.to(device)

            opt.zero_grad()
            loss1 = F.cross_entropy(outputs, targets)
            loss1.backward()
            opt.step()

            total_loss1 += loss1.item()

        for i, (data, publish, targets, pos) in enumerate(train_iter_many):
            targets, pos = targets.to(device), pos.to(device)
            outputs = model(data.to(device), publish.to(device), pos, 'many', targets)

            opt.zero_grad()
            targets = targets[pos - 1]
            loss2 = F.cross_entropy(outputs, targets)
            loss2.backward()
            opt.step()

            total_loss2 += loss2.item()
            # del data, publish, targets,outputs
            # torch.cuda.empty_cache()

        total_loss1 = total_loss1 / len(train_iter_one)
        total_loss2 = total_loss2 / len(train_iter_many)

        t_b = time.time()
        msg = 'Train Loss: {0:>5.4},  Time: {1:>6.2}'
        print(msg.format(total_loss1 + total_loss2, t_b - t_a))

        # 验证-------------------------------
        model.eval()
        t_a = time.time()
        total_loss1, total_loss2, total_f1, one_f1, many_f1 = 0, 0, 0, 0, 0
        with torch.no_grad():
            for i, (data, publish, targets) in enumerate(dev_iter_one):
                outputs = model(data.to(device), publish.to(device), 'one')
                targets = targets.to(device)
                # loss = F.cross_entropy(outputs, targets)

                true = targets.data.cpu()
                label = torch.max(outputs.data, 1)[1].cpu()
                # total_loss1 += float(loss.item())
                one_f1 += f1_score(true, label, average='macro')

        true_list, label_list = torch.tensor([]), torch.tensor([])
        with torch.no_grad():
            for i, (data, publish, targets, pos) in enumerate(dev_iter_many):
                targets, pos = targets.to(device), pos.to(device)
                outputs = model(data.to(device), publish.to(device), pos, 'many', targets)
                targets = targets[pos - 1]
                # loss = F.cross_entropy(outputs, targets)

                true = targets.data.detach().cpu()
                label = torch.max(outputs.data, 1)[1].detach().cpu()
                true_list = torch.cat((true_list, true), dim=0)
                label_list = torch.cat((label_list, label), dim=0)
                # total_loss2 += float(loss.item())
                # many_f1 += f1_score(true, label, average='macro')
        many_f1 = f1_score(true_list, label_list, average='macro')
        all_f1 = f1_score(true_list, label_list, average=None)
        print(all_f1)

        print('one f1 {}, many f1 {}'.format(one_f1 / len(dev_iter_one), many_f1))
        total_f1 = (one_f1 / len(dev_iter_one) + many_f1) / 2
        # total_loss = total_loss1 / len(dev_iter_one) + total_loss2 / len(dev_iter_many)

        t_b = time.time()
        # msg = 'Eval Loss: {0:>5.4},  Eval f1: {1:>6.2%},  Time: {2:>7.2}'
        msg = 'Eval f1: {0:>6.2%},  Time: {1:>7.2}'
        # print(msg.format(total_loss, total_f1, t_b - t_a))
        print(msg.format(total_f1, t_b - t_a))

        if total_f1 > best_f1:
            best_f1 = total_f1
            torch.save(model.state_dict(), './model/vnews_stock.pkl')
            print('效果提升,保存最优参数')
            cnt = 0
        else:
            cnt += 1
            if cnt == 2:
                print('模型已无提升停止训练,验证集最高f1:%.2f' % (100 * best_f1))
                stop_flag = True
                break

    if stop_flag:
        pass
    else:
        print('训练结束,验证集最高f1:%.2f' % (100 * best_f1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--EPOCHS', type=int, default=3)
    parser.add_argument('--BATCH_SIZE', type=int, default=64)
    parser.add_argument('--LEARNING_RATE', type=float, default=1e-5)
    parser.add_argument('--MAX_LEN', default=140)

    args = parser.parse_args()
    seed_torch(args.seed)

    MODEL_CKPT = 'hfl/chinese-bert-wwm-ext'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)


    def dynamic_collate_1(data):
        """Custom data collator for dynamic padding."""
        inputs = [d for d, k, l in data]
        labels = torch.stack([l for d, k, l in data], dim=0)

        inputs = tokenizer.pad(inputs, return_tensors='pt')

        res = []
        for d, k, l in data:
            res.append(k)
        keywords = torch.tensor(res, dtype=torch.long)

        return inputs, keywords, labels


    # def dynamic_collate_2(data):
    #     """Custom data collator for dynamic padding."""
    #     inputs = [d for d, k in data]
    #
    #     inputs = tokenizer.pad(inputs, return_tensors='pt')
    #
    #     res = []
    #     for d, k in data:
    #         res.append(k)
    #     keywords = torch.tensor(res, dtype=torch.long)
    #     # keywords = torch.stack([[k] * Max_len for d, k, l in data], dim=0)
    #     return inputs, keywords

    def dynamic_collate_3(data):

        inputs_list, keyword_list, label_list, pos_list = [], [], [], [0]
        for inputs, k, l in data:
            inputs_list.extend(inputs)
            keyword_list.extend(k)
            num = len(k)
            pos_list.append(num + pos_list[-1])
            label_list.extend([l] * num)

        inputs_list = tokenizer.pad(inputs_list, return_tensors='pt')
        label_list = torch.tensor(label_list, dtype=torch.long)
        keyword_list = torch.tensor(keyword_list, dtype=torch.long)
        pos_list = torch.tensor(pos_list, dtype=torch.long)

        return inputs_list, keyword_list, label_list, pos_list[1:]


    # def dynamic_collate_4(data):
    #     data = data[0]
    #     inputs = [d for d, k in data]
    #
    #     inputs = tokenizer.pad(inputs, return_tensors='pt')
    #
    #     res = []
    #     for d, k in data:
    #         res.append(k)
    #     keywords = torch.tensor(res, dtype=torch.long)
    #     # keywords = torch.stack([[k] * Max_len for d, k, l in data], dim=0)
    #     return inputs, keywords

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print("device {}".format(device))

    # 分层训练
    df = pd.read_parquet('./filter/vnews_stock.parquet')
    # df = pd.read_parquet('./filter/vnews_stock_filter.parquet')
    year = 2022
    month = 1

    while 1:
        if year == 2022 and month + 3 > 11:
            break
        start = str(year) + '-' + str(month).rjust(2, '0')
        train_month = [
            str(year) + '-' + str(x).rjust(2, '0') if x <= 12 else str(year + 1) + '-' + str(x - 12).rjust(2, '0') \
            for x in range(month, month + 3)]
        eval_month = [
            str(year) + '-' + str(x).rjust(2, '0') if x <= 12 else str(year + 1) + '-' + str(x - 12).rjust(2, '0') \
            for x in range(month + 3, month + 4)]

        train_month = '|'.join(train_month)
        eval_month = '|'.join(eval_month)
        # df=df[df['date']]
        tmp = df.copy()
        train_df = tmp[tmp['date'].str.contains(train_month)]
        train_df = train_df[['publish', 'content', 'label', 'len']]
        eval_df = tmp[tmp['date'].str.contains(eval_month)]
        eval_df = eval_df[['publish', 'content', 'label', 'len']]

        train_df.reset_index(drop=True, inplace=True)
        eval_df.reset_index(drop=True, inplace=True)
        print('-' * 10)
        print('start date:{} ,train data:{} ,eval data:{}'.format(start, len(train_df), len(eval_df)))

        label_list1 = [x for x in train_df['label'].values]
        label_list1 = collections.Counter(label_list1)
        print(label_list1)

        label_list2 = [x for x in eval_df['label'].values]
        label_list2 = collections.Counter(label_list2)
        print(label_list2)

        # 根据该条股票的评论新闻数目，使用不同的loader进行装载
        train_df1 = train_df[train_df['len'] == 1]
        train_df2 = train_df.drop(train_df1.index)
        eval_df1 = eval_df[eval_df['len'] == 1]
        eval_df2 = eval_df.drop(eval_df1.index)
        print(len(train_df1), len(train_df2))
        print(len(eval_df1), len(eval_df2))

        train_df1.reset_index(drop=True, inplace=True)
        train_df2.reset_index(drop=True, inplace=True)
        eval_df1.reset_index(drop=True, inplace=True)
        eval_df2.reset_index(drop=True, inplace=True)

        # 一只股票对应一条新闻用正常方式装载
        train_set1 = StockDataset_one(train_df1, tokenizer, args.MAX_LEN)
        eval_set1 = StockDataset_one(eval_df1, tokenizer, args.MAX_LEN)
        train_loader1 = DataLoader(train_set1, batch_size=args.BATCH_SIZE, shuffle=True, collate_fn=dynamic_collate_1,
                                   drop_last=True)
        val_loader1 = DataLoader(eval_set1, batch_size=2 * args.BATCH_SIZE, shuffle=False, collate_fn=dynamic_collate_1,
                                 drop_last=True)

        # 一只股票对应多条新闻用一个batch方式装载
        train_set2 = StockDataset_many(train_df2, tokenizer, args.MAX_LEN)
        eval_set2 = StockDataset_many(eval_df2, tokenizer, args.MAX_LEN)
        train_loader2 = DataLoader(train_set2, batch_size=15, shuffle=True, collate_fn=dynamic_collate_3,
                                   drop_last=True)
        val_loader2 = DataLoader(eval_set2, batch_size=15, shuffle=False, collate_fn=dynamic_collate_3, drop_last=True)

        model = News_Stock()
        # if os.path.exists('./model/vnews_stock.pkl'):
        #     model.load_state_dict(torch.load('./model/vnews_stock.pkl'))
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.LEARNING_RATE, weight_decay=0.5)

        train(model, train_loader1, train_loader2, val_loader1, val_loader2, args, optimizer)
        # del tmp, train_df, eval_df, train_data1, train_data2, eval_data1, eval_data2
        # torch.cuda.empty_cache()
        break

        month += 1
        if month > 12:
            month -= 12
            year += 1
