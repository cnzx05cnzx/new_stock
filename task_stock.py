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


class StockPredictDataset(Dataset):

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
        text = self.text[index]
        publish = self.publish[index]

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


class News_Stock(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_CKPT)
        self.embedding = nn.Embedding(28592, 256)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
            # nn.BatchNorm1d(64),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(64, 2)
        )

    def forward(self, inputs, publish):
        bert_output = self.bert(**inputs)
        hidden_state = bert_output.last_hidden_state
        pooled_out = hidden_state[:, 0]
        key_bedding = self.embedding(publish)
        all_out = torch.cat([pooled_out, key_bedding], dim=1)
        logits = self.classifier(all_out)
        return logits


def train(model, train_iter, dev_iter, args, opt):

    best_f1 = float(0)
    cnt = 0  # 记录多久没有模型效果提升
    stop_flag = False  # 早停标签

    for epoch in range(args.EPOCHS):
        print('Epoch [{}/{}]'.format(epoch + 1, args.EPOCHS))

        # 训练-------------------------------
        model.train()
        t_a = time.time()
        total_loss = 0

        for i, (data, publish, targets) in enumerate(train_iter):
            outputs = model(data.to(device), publish.to(device))
            targets = targets.to(device)

            opt.zero_grad()
            loss = F.cross_entropy(outputs, targets)
            # flood=(loss-)
            loss.backward()
            opt.step()

            total_loss += float(loss.item())

        total_loss = total_loss / len(train_iter)

        t_b = time.time()
        msg = 'Train Loss: {0:>5.4},  Time: {1:>6.2}'
        print(msg.format(total_loss, t_b - t_a))

        # 验证-------------------------------
        model.eval()
        t_a = time.time()
        total_loss, total_f1 = 0, 0

        for i, (data, publish, targets) in enumerate(dev_iter):
            with torch.no_grad():
                outputs = model(data.to(device), publish.to(device))
                targets = targets.to(device)
                loss = F.cross_entropy(outputs, targets)

            true = targets.data.cpu()
            label = torch.max(outputs.data, 1)[1].cpu()
            total_loss += float(loss.item())
            total_f1 += f1_score(true, label, average='macro')

        total_f1 = total_f1 / len(dev_iter)
        total_loss = total_loss / len(dev_iter)

        t_b = time.time()
        msg = 'Eval Loss: {0:>5.4},  Eval f1: {1:>6.2%},  Time: {2:>7.2}'
        print(msg.format(total_loss, total_f1, t_b - t_a))

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
    parser.add_argument('--BATCH_SIZE', type=int, default=128)
    parser.add_argument('--LEARNING_RATE', type=float, default=2e-5)
    parser.add_argument('--MAX_LEN', default=150)

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


    def dynamic_collate_2(data):
        """Custom data collator for dynamic padding."""
        inputs = [d for d, k in data]

        inputs = tokenizer.pad(inputs, return_tensors='pt')

        res = []
        for d, k in data:
            res.append(k)
        keywords = torch.tensor(res, dtype=torch.long)
        # keywords = torch.stack([[k] * Max_len for d, k, l in data], dim=0)
        return inputs, keywords


    train_params = {'batch_size': args.BATCH_SIZE,
                    'shuffle': False,
                    'collate_fn': dynamic_collate_1}

    eval_params = {'batch_size': args.BATCH_SIZE * 2,
                   'shuffle': False,
                   'collate_fn': dynamic_collate_1}

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print("device {}".format(device))

    # 分层训练
    df = pd.read_parquet('./filter/vnews_stock.parquet')
    # df = pd.read_parquet('./filter/vnews_stock_filter.parquet')
    year = 2019
    month = 1
    while 1:
        if year == 2022 and month + 3 > 12:
            break
        pos = str(year) + '-' + str(month).rjust(2, '0')
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
        train_df = train_df[['publish', 'content', 'label']]
        eval_df = tmp[tmp['date'].str.contains(eval_month)]
        eval_df = eval_df[['publish', 'content', 'label']]


        train_df.reset_index(drop=True, inplace=True)
        eval_df.reset_index(drop=True, inplace=True)
        print('-' * 10)
        print('start date:{} ,train data:{} ,eval data:{}'.format(pos, len(train_df), len(eval_df)))

        label_list1=[x for x in train_df['label'].values]
        label_list1=collections.Counter(label_list1)
        print(label_list1)


        label_list2=[x for x in eval_df['label'].values]
        label_list2=collections.Counter(label_list2)
        print(label_list2)

        train_set = StockPredictDataset(train_df, tokenizer, args.MAX_LEN)
        eval_set = StockPredictDataset(eval_df, tokenizer, args.MAX_LEN)

        train_loader = DataLoader(train_set, **train_params)
        val_loader = DataLoader(eval_set, **eval_params)

        model = News_Stock()
        if os.path.exists('./model/vnews_stock.pkl'):
            model.load_state_dict(torch.load('./model/vnews_stock.pkl'))
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.LEARNING_RATE, weight_decay=0.5)


        train(model, train_loader, val_loader, args, optimizer)
        # break

        month += 1
        if month > 12:
            month -= 12
            year += 1
