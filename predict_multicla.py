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


def predict(model, test_iter):
    # 测试-------------------------------
    model.load_state_dict(torch.load('./model/vnews_tag.pkl'))
    model.to(device)

    model.eval()
    res = []
    for i, (data) in enumerate(test_iter):
        with torch.no_grad():
            outputs = model(data.to(device))
        predict = torch.sigmoid(outputs).cpu().detach()
        predict = (predict > 0.5).int()
        res.append(predict)
    return res


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

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("device {}".format(device))
    MODEL_CKPT = 'hfl/chinese-bert-wwm-ext'

    model = Muti_cla()

    # 预测数据
    test_data = pd.read_csv('./filter/tag_test.csv')

    print("Num. samples:", len(test_data))


    tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT, do_lower_case=True)
    test_set = MultiLabelDataset(test_data, tokenizer, args.MAX_LEN, new_data=True)


    def dynamic_collate_2(data):
        """Custom data collator for dynamic padding."""
        inputs = [d for d in data]
        inputs = tokenizer.pad(inputs, return_tensors='pt')

        return inputs


    test_params = {'batch_size': args.BATCH_SIZE * 2,
                   'shuffle': False,
                   'collate_fn': dynamic_collate_2}

    test_loader = DataLoader(test_set, **test_params)

    all_test_pred = predict(model, test_loader)
    test_data['stock'] = torch.cat(all_test_pred).cpu().numpy().tolist()


    def fun(x):
        res = []
        for index, r in enumerate(x):
            if r == 1:
                res.append(index)
        return res


    test_data['stock'] = test_data['stock'].apply(lambda x: fun(x))
    test_data.to_csv('./filter/tag_res.csv', index=False)
