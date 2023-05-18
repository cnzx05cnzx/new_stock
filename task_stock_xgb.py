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

os.environ['CUDA_VISIBLE_DEVICES'] = '4'


def seed_torch(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = True


def XGB_model(X_train, Y_train, X_test, Y_test, day):
    t_a = time.time()
    param_dist = {
        'max_depth': [3, 4, 5, 6, 7, 9, 12, 15],
        'learning_rate': [0.01, 0.025, 0.05, 0.075],
        'subsample': [0.8, 0.85, 0.9, 0.95],
    }
    print(param_dist)

    xgb_param = {'booster': 'gbtree',
                 'objective': 'multi:softmax',
                 'num_class': 3,
                 'gamma': 0.1,
                 'max_depth': 5,
                 'subsample': 0.8,
                 'colsample_bytree': 0.95,
                 'learning_rate': 0.01,
                 'seed': 1,
                 'nthread': 4,
                 'tree_method': 'gpu_hist',
                 'gpu_id': 0}

    bml = XGBClassifier(**xgb_param)

    gsearch = GridSearchCV(estimator=bml,  # 交叉验证和网格搜索
                           param_grid=param_dist,
                           cv=5,
                           scoring='accuracy',
                           n_jobs=4
                           )
    gsearch.fit(X_train, Y_train)

    best_estimator = gsearch.best_estimator_  # 通过搜索选择的估计器，即在左侧数据上给出最高分数（或指定的最小损失）的估计器。 如果refit = False，则不可用

    print('max_depth_min_child_weight')
    print('gsearch1.best_params_', gsearch.best_params_)  # 最佳结果的参数设置
    print('gsearch1.best_score_', gsearch.best_score_)  # best_estimator的分数

    score = gsearch.score(X_test, Y_test)
    print('预测准确度：\n', score)

    best_model = gsearch.best_estimator_
    # 保存最佳模型

    day = day.strftime("%Y-%m-%d")
    path = './model/vnews_stock_xgboost_{}.pickle'.format(day)
    with open(path, 'wb') as f:
        pickle.dump(best_model, f)

    predict_Y = best_model.predict(X_test)
    F1_SCORE = metrics.f1_score(Y_test, predict_Y, average='macro')
    print('预测F1：\n', F1_SCORE)
    CM = confusion_matrix(Y_test, predict_Y)
    print('混淆矩阵', CM)
    t_b = time.time()
    print('cost time {} min'.format((t_b - t_a) / 60))
    return gsearch.best_score_, F1_SCORE, best_estimator


def judge(x, a, b):
    if x <= a:
        return 0
    elif x <= b:
        return 1
    else:
        return 2


# 29.17 1e-6
# 30.39(去除) focal > cross
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--EPOCHS', type=int, default=3)
    parser.add_argument('--BATCH_SIZE', type=int, default=64)
    parser.add_argument('--LEARNING_RATE', type=float, default=1e-5)
    parser.add_argument('--MAX_LEN', default=140)

    args = parser.parse_args()
    seed_torch(args.seed)

    # 分层训练
    df = pd.read_parquet('./filter/vnews_stock_merge2.parquet')
    # print(len(df))
    df['date'] = pd.to_datetime(df['date']).dt.date
    # print(df[-100:].head())

    start_date = datetime.date(2022, 5, 17)
    end_date = datetime.date(2022, 5, 30)
    current_date = start_date
    while current_date <= end_date:
        print('*' * 20)
        print('test day is {}'.format(current_date))
        train_day = current_date - datetime.timedelta(days=90)

        # 加载测试机
        tmp = df.copy()
        mask = (tmp['date'] == current_date)
        tmp_data = tmp.loc[mask]
        tmp_data = tmp_data[['stock_id', 'frquent', 'ref_pct']]
        if len(tmp_data) > 1:
            # 加载训练集与验证集
            tmp = df.copy()
            mask = (tmp['date'] >= train_day) & (tmp['date'] < current_date)
            tmp_data = tmp.loc[mask]
            tmp_data = tmp_data[['stock_id', 'frquent', 'ref_pct']]

            q10 = tmp_data.quantile(.3, numeric_only=True).ref_pct
            q90 = tmp_data.quantile(.7, numeric_only=True).ref_pct
            tmp_data['label'] = tmp_data['ref_pct'].apply(lambda x: judge(x, q10, q90))
            tmp_data.reset_index(drop=True, inplace=True)

            lbl = preprocessing.LabelEncoder()
            tmp_data['label'] = lbl.fit_transform(tmp_data['label'].astype(str))  # 将提示的包含错误数据类型这一列进行转换

            x = tmp_data['frquent'].values.tolist()  # 将列的元素直接转换为列表
            y = tmp_data['label'].values.tolist()

            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=args.seed)

            print(len(x_train), len(x_val))

            XGB_model(x_train, y_train, x_val, y_val, current_date)

        # break

        current_date += datetime.timedelta(days=1)
