import argparse
import collections
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
import random
from sklearn.metrics import f1_score
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4'


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
        text = '新闻预测股票会[MASK][MASK]。' + str(self.text[index])

        # text = '大风洪水暴雨地震冰雹'
        encoded_sent = tokenizer.encode_plus(
            text=text,  # 预处理语句
            add_special_tokens=True,  # 加 [CLS] 和 [SEP]
            max_length=self.max_len,  # 截断或者填充的最大长度
            padding='max_length',  # 填充为最大长度，这里的padding在之间可以直接用pad_to_max但是版本更新之后弃用了，老版本什么都没有，可以尝试用extend方法
            return_attention_mask=True,  # 返回 attention mask
            truncation=True
        )

        # 把输出加到列表里面
        input_ids = encoded_sent.get('input_ids')
        attention_masks = encoded_sent.get('attention_mask')
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_masks = torch.tensor(attention_masks, dtype=torch.long)
        if not self.new_data:
            labels = torch.tensor(self.targets[index], dtype=torch.long)
            return input_ids, attention_masks, labels

        return input_ids, attention_masks


def train(model, train_iter, dev_iter, args, opt):
    criterion = torch.nn.CrossEntropyLoss()
    mask1 = 7
    mask2 = 8
    best_f1 = float(0)
    cnt = 0  # 记录多久没有模型效果提升
    stop_flag = False  # 早停标签
    for epoch in range(args.EPOCHS):
        print('Epoch [{}/{}]'.format(epoch + 1, args.EPOCHS))

        # 训练-------------------------------
        model.train()
        t_a = time.time()
        total_loss = 0
        for i, (inputs, masks, targets) in enumerate(train_iter):
            outputs = model(input_ids=inputs.to(device), attention_mask=masks.to(device))
            targets = targets.to(device)

            opt.zero_grad()
            logits = outputs.logits

            pred1 = logits[:, mask1, [678, 7448, 677]]
            pred2 = logits[:, mask2, [6649, 5782, 3885]]
            loss1 = criterion(pred1, targets)
            loss2 = criterion(pred2, targets)
            loss = loss1 + loss2

            loss.backward()
            opt.step()
            total_loss += float(loss.item())
        total_loss = total_loss / len(train_iter)

        t_b = time.time()
        msg = 'Train Loss: {0:>5.4},  Time: {1:>6.2}'
        print(msg.format(total_loss, t_b - t_a))

        # 验证-------------------------------
        model.eval()
        total_loss, total_f1 = 0, 0
        with torch.no_grad():
            for i, (inputs, masks, targets) in enumerate(dev_iter):
                outputs = model(input_ids=inputs.to(device), attention_mask=masks.to(device))
                targets = targets.to(device)

                logits = outputs.logits

                pred1 = logits[:, mask1, [678, 7448, 677]]
                pred2 = logits[:, mask2, [6649, 5782, 3885]]
                loss1 = criterion(pred1, targets)
                loss2 = criterion(pred2, targets)
                loss = loss1 + loss2

                logit1 = pred1.detach().cpu().numpy()
                logit1 = np.argmax(logit1, axis=-1)
                logit2 = pred2.detach().cpu().numpy()
                logit2 = np.argmax(logit2, axis=-1)

                label = [a if a == b else 1 for a, b in zip(logit1, logit2)]

                total_loss += float(loss.item())
                true = targets.data.cpu()
                total_f1 += f1_score(true, label, average='macro')

            t_b = time.time()
            total_f1 = total_f1 / len(dev_iter)
            msg = 'Eval Loss: {0:>5.4},  Eval f1: {1:>6.2%},  Time: {2:>7.2}'
            print(msg.format(total_loss / len(dev_iter), total_f1, t_b - t_a))

            if total_f1 > best_f1:
                best_f1 = total_f1
                torch.save(model.state_dict(), './model/vnews_stock_prompt.pkl')
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


def predict(model, test_iter):
    # 测试-------------------------------
    model.load_state_dict(torch.load('./saved_dict/bert.pkl'))
    model.eval()
    res = []
    for i, (data) in enumerate(test_iter):
        with torch.no_grad():
            outputs = model(data.cuda())

        predict = torch.max(outputs.data, 1)[1].cpu()
        res.append(predict)
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--EPOCHS', type=int, default=5)
    parser.add_argument('--BATCH_SIZE', type=int, default=64)
    parser.add_argument('--LEARNING_RATE', type=float, default=2e-4)
    parser.add_argument('--MAX_LEN', default=150)

    args = parser.parse_args()
    seed_torch(args.seed)

    MODEL_CKPT = 'hfl/chinese-bert-wwm-ext'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)

    train_params = {'batch_size': args.BATCH_SIZE,
                    'shuffle': True}

    eval_params = {'batch_size': args.BATCH_SIZE * 2,
                   'shuffle': False}

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print("device {}".format(device))

    # 分层训练
    df = pd.read_parquet('./filter/vnews_stock.parquet')
    # df = pd.read_parquet('./filter/vnews_stock_filter.parquet')
    year = 2020
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

        label_list1 = [x for x in train_df['label'].values]
        label_list1 = collections.Counter(label_list1)
        print(label_list1)

        label_list2 = [x for x in eval_df['label'].values]
        label_list2 = collections.Counter(label_list2)
        print(label_list2)

        train_set = StockPredictDataset(train_df, tokenizer, args.MAX_LEN)
        eval_set = StockPredictDataset(eval_df, tokenizer, args.MAX_LEN)

        train_loader = DataLoader(train_set, **train_params)
        val_loader = DataLoader(eval_set, **eval_params)

        model = AutoModelForMaskedLM.from_pretrained(MODEL_CKPT)
        # if os.path.exists('./model/vnews_stock.pkl'):
        #     model.load_state_dict(torch.load('./model/vnews_stock.pkl'))
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.LEARNING_RATE, weight_decay=0.5)
        train(model, train_loader, val_loader, args, optimizer)
        break

        month += 1
        if month > 12:
            month -= 12
            year += 1
