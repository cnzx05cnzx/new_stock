# import argparse
# import re
# import os
# import string
# import time
#
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from sklearn import metrics
# from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
# from transformers import AutoTokenizer, AutoModel
# import random
# import torch.nn.functional as F
# from collections import Counter
#
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
#
#
# def seed_torch(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.enabled = True
#
#
# class StockPredictDataset(Dataset):
#
#     def __init__(self, dataframe, tokenizer, max_len, new_data=False):
#         self.data = dataframe
#         self.tokenizer = tokenizer
#         self.text = dataframe.content
#         self.publish = dataframe.publish
#         self.new_data = new_data
#         self.max_len = max_len
#
#         if not new_data:
#             self.targets = self.data.label
#
#     def __len__(self):
#         return len(self.text)
#
#     def __getitem__(self, index):
#         text = self.text[index]
#         publish = self.publish[index]
#
#         inputs = self.tokenizer(
#             text,
#             truncation=True,
#             padding='max_length' if self.new_data else False,
#             max_length=self.max_len,
#             return_tensors="pt"
#         )
#         inputs = {k: v.squeeze() for k, v in inputs.items()}
#
#         if not self.new_data:
#             labels = torch.tensor(self.targets[index], dtype=torch.long)
#             return inputs, publish, labels
#
#         return inputs, publish
#
#
# class News_Stock(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.bert = AutoModel.from_pretrained(MODEL_CKPT)
#         self.embedding = nn.Embedding(28592, 256)
#         self.classifier = nn.Sequential(
#             nn.Linear(1024, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, 3)
#         )
#
#     def forward(self, inputs, publish):
#         bert_output = self.bert(**inputs)
#         hidden_state = bert_output.last_hidden_state
#         pooled_out = hidden_state[:, 0]
#         key_bedding = self.embedding(publish)
#         all_out = torch.cat([pooled_out, key_bedding], dim=1)
#         logits = self.classifier(all_out)
#         return logits
#
#
# def predict(model, test_iter):
#     # 测试-------------------------------
#     model.load_state_dict(torch.load('./model/vnews_stock.pkl'))
#     model.to(device)
#
#     model.eval()
#     res = []
#     for i, (data, publish) in enumerate(test_iter):
#         with torch.no_grad():
#             outputs = model(data.to(device), publish.to(device))
#         predict = torch.max(outputs.data, 1)[1].cpu()
#         res.append(predict)
#     return res
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--seed', type=int, default=1)
#     parser.add_argument('--model_dir',
#                         help='Path to the model dir', default='model')
#     parser.add_argument('--EPOCHS', type=int, default=7)
#     parser.add_argument('--BATCH_SIZE', type=int, default=128)
#     parser.add_argument('--LEARNING_RATE', type=float, default=1e-5)
#     parser.add_argument('--MAX_LEN', default=150)
#
#     args = parser.parse_args()
#     seed_torch(args.seed)
#
#     device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#     print("device {}".format(device))
#     MODEL_CKPT = 'hfl/chinese-bert-wwm-ext'
#
#     model = News_Stock()
#
#     # 预测数据
#     test_data = pd.read_csv('./filter/stock_test.csv')
#
#     print("Num. samples:", len(test_data))
#
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT, do_lower_case=True)
#     test_set = StockPredictDataset(test_data, tokenizer, args.MAX_LEN, new_data=True)
#
#
#     def dynamic_collate_2(data):
#         """Custom data collator for dynamic padding."""
#         inputs = [d for d, k in data]
#
#         inputs = tokenizer.pad(inputs, return_tensors='pt')
#
#         res = []
#         for d, k in data:
#             res.append(k)
#         keywords = torch.tensor(res, dtype=torch.long)
#         # keywords = torch.stack([[k] * Max_len for d, k, l in data], dim=0)
#         return inputs, keywords
#
#
#     test_params = {'batch_size': args.BATCH_SIZE * 2,
#                    'shuffle': False,
#                    'collate_fn': dynamic_collate_2}
#
#     test_loader = DataLoader(test_set, **test_params)
#
#     all_test_pred = predict(model, test_loader)
#     test_data['label'] = all_test_pred[:].cpu()
#     test_data.to_csv('./filter/stock_res.csv', index=False)
#
import datetime

date = datetime.date(2021, 2, 3)
formatted_date = date.strftime('%Y-%m-%d')
print(formatted_date) # 输出 2021-02-03
