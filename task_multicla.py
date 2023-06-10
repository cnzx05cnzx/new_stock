import argparse
import re
import os
import string
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModel
import random
import torch.nn.functional as F
from collections import Counter

os.environ['CUDA_VISIBLE_DEVICES'] = '6'


def seed_torch(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


class MultiLabelDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len, new_data=False):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.text = dataframe.NEWS_SUMMARY
        self.max_len = max_len
        self.new_data = new_data

        if not new_data:
            self.label = self.data.label

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        # keyword = torch.tensor(keyword, dtype=torch.long)

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length' if self.new_data else False,
            max_length=self.max_len,
            return_tensors="pt"
        )
        inputs = {k: v.squeeze() for k, v in inputs.items()}

        if not self.new_data:
            label = self.label[index]
            label = [int(x) for x in list(label)]
            labels = torch.tensor(label, dtype=torch.float)
            return inputs, labels

        return inputs


class Muti_cla(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_CKPT)
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 2810)
        )

    def forward(self, inputs):
        bert_output = self.bert(**inputs)
        hidden_state = bert_output.last_hidden_state
        pooled_out = hidden_state[:, 0]
        logits = self.classifier(pooled_out)
        return logits


def accuracy_multi(inp, targ, thresh=0.5, sigmoid=True):
    """An accuracy metric for multi-label problems."""
    if sigmoid:
        inp = inp.sigmoid()
    return ((inp > thresh) == targ.bool()).float().mean()


def train(model, train_iter, dev_iter, args, opt, lr_s):
    optimizer = opt
    get_loss = nn.BCEWithLogitsLoss()

    best_f1 = float(0)
    cnt = 0  # 记录多久没有模型效果提升
    stop_flag = False  # 早停标签

    for epoch in range(args.EPOCHS):
        print('Epoch [{}/{}]'.format(epoch + 1, args.EPOCHS))

        # 训练-------------------------------
        model.train()
        t_a = time.time()
        total_loss, total_acc = 0, 0

        for i, (data, targets) in enumerate(train_iter):
            outputs = model(data.to(device))
            targets = targets.to(device)

            model.zero_grad()
            loss = get_loss(outputs, targets)

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        lr_s.step()

        total_loss = total_loss / len(train_iter)

        t_b = time.time()
        msg = 'Train Loss: {0:>5.2},  Time: {1:>7.2}'
        print(msg.format(total_loss, t_b - t_a))

        # 验证-------------------------------

        model.eval()
        t_a = time.time()
        total_loss, total_p, total_f1 = 0, 0, 0
        # fin_targets, fin_outputs = [], []
        for i, (data, targets) in enumerate(dev_iter):
            with torch.no_grad():
                outputs = model(data.to(device))
                targets = targets.to(device)
                loss = get_loss(outputs, targets)

            total_loss += float(loss.item())

            fin_targets = targets.cpu().detach().view(1, -1).squeeze()
            fin_outputs = torch.sigmoid(outputs).cpu().detach().view(1, -1).squeeze()
            fin_outputs = (fin_outputs > 0.5).int()
            total_f1 += metrics.f1_score(fin_targets, fin_outputs)
            total_p += metrics.precision_score(fin_targets, fin_outputs)
            # print(1)

        total_loss = total_loss / len(dev_iter)
        total_f1 = total_f1 / len(dev_iter)
        total_p = total_p / len(dev_iter)

        t_b = time.time()
        msg = 'Eval Loss: {0:>5.2},  Eval f1: {1:>6.2%},Eval p: {2:>6.2%},  Time: {3:>7.2}'
        print(msg.format(total_loss, total_f1, total_p, t_b - t_a))

        if total_f1 > best_f1:
            best_f1 = total_f1
            torch.save(model.state_dict(), './model/vnews_tag.pkl')
            print('效果提升,保存最优参数')
            cnt = 0
        else:
            cnt += 1
            if cnt > 2:
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
    parser.add_argument('--model_dir',
                        help='Path to the model dir', default='model')
    parser.add_argument('--EPOCHS', type=int, default=7)
    parser.add_argument('--BATCH_SIZE', type=int, default=128)
    parser.add_argument('--LEARNING_RATE', type=float, default=1e-5)
    parser.add_argument('--MAX_LEN', default=150)

    args = parser.parse_args()
    seed_torch(args.seed)

    MODEL_CKPT = 'hfl/chinese-bert-wwm-ext'

    df = pd.read_parquet('./filter/vnews_tag.parquet')
    df = df[['NEWS_SUMMARY', 'label']]

    split_size = 0.8
    train_df = df.sample(frac=split_size, random_state=args.seed)
    val_df = df.drop(train_df.index).reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)

    print("Training Dataset: {}".format(train_df.shape))
    print("Validation Dataset: {}".format(val_df.shape))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)
    train_set = MultiLabelDataset(train_df, tokenizer, args.MAX_LEN)
    val_set = MultiLabelDataset(val_df, tokenizer, args.MAX_LEN)


    def dynamic_collate_1(data):
        """Custom data collator for dynamic padding."""
        inputs = [d for d, l in data]
        labels = torch.stack([l for d, l in data], dim=0)
        inputs = tokenizer.pad(inputs, return_tensors='pt')

        # keywords = torch.stack([[k] * Max_len for d, k, l in data], dim=0)
        return inputs, labels


    def dynamic_collate_2(data):
        """Custom data collator for dynamic padding."""
        inputs = [d for d in data]
        inputs = tokenizer.pad(inputs, return_tensors='pt')

        return inputs


    train_params = {'batch_size': args.BATCH_SIZE,
                    'shuffle': False,
                    'collate_fn': dynamic_collate_1}

    val_params = {'batch_size': args.BATCH_SIZE,
                  'shuffle': False,
                  'collate_fn': dynamic_collate_1}

    train_loader = DataLoader(train_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("device {}".format(device))

    model = Muti_cla()
    # model.load_state_dict(torch.load('./model/vnews_tag.pkl'))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.LEARNING_RATE)
    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.2)

    train(model, train_loader, val_loader, args, optimizer, lr_sched)
    # f1 63 p 90

    # 预测数据
    # test_data = pd.read_csv('./disasters/test.csv')
    #
    # print("Num. samples:", len(test_data))
    #
    # test_data["keyword"] = test_data["keyword"].fillna("none")
    # test_data["location"] = test_data["location"].fillna("none")
    # test_data['comment'] = test_data['text']
    #
    # label_columns = ["location", "text"]
    # test_data.drop(label_columns, inplace=True, axis=1)
    #
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT, do_lower_case=True)
    # test_set = MultiLabelDataset(test_data, tokenizer, args.MAX_LEN, new_data=True)
    #
    # test_params = {'batch_size': args.BATCH_SIZE * 2,
    #                'shuffle': False,
    #                'collate_fn': dynamic_collate_2}
    #
    # test_loader = DataLoader(test_set, **test_params)
    #
    # all_test_pred = predict(model, test_loader)
    # all_test_pred = torch.cat(all_test_pred)
    # test_data['target'] = all_test_pred[:].cpu()
    # submit_df = test_data[['id', 'target']]
    # submit_df.to_csv('./disasters/submission.csv', index=False)
