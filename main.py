import datetime
import os

os.environ['JAVA_HOME'] = '/home/zhuzhicheng/java/jdk-20/'
import pandas as pd
from pyhanlp import *

from sklearn.feature_extraction.text import TfidfVectorizer

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

HanLP.Config.ShowTermNature = False
# 设置环境变量
from tqdm import tqdm
import numpy as np
from xgboost import XGBClassifier



HEADLINE_LEN = 100
ABSTRACT_LEN = 500


# 加载停用词词典
def load_file(filename):
    with open(filename, 'r', encoding="utf-8") as f:
        contents = f.readlines()
    result = []
    for content in contents:
        result.append(content.strip())
    return result


# 去停用词
def remove_stop_words(text, dic):
    result = []
    for k in text:
        if k.word not in dic:
            result.append(k.word)
    return result


# def extract(result,weight,word):
#     length = len(weight)
#     data = [[] for i in range(len(weight))]
#
#     for i in range(length):
#         list = []
#         rank = []
#         dict = {}
#         for j in range(len(word)):
#             if weight[i][j] != 0:
#                 if word[j] not in dict.keys():
#                     dict[word[j]] = 0
#                 dict[word[j]] = weight[i][j]
#                 list.append(word[j])
#                 rank.append(weight[i][j])
#
#         #根据tf-idf进行排序
#         tfidf_sorted = sorted(dict.items(),
#                                           key=lambda x: x[1], reverse=True)
#         tfidf_dict_sorted = {}
#         tfidf_filter = {}
#         for rank, (stu_key, stu_val) in enumerate(
#                 tfidf_sorted, 1):
#             tfidf_dict_sorted[stu_key] = (rank, stu_val)     # 重新构造带有排名的排序后的分词字典tfidf_dict_sorted
#
#             if rank<len(dict)/2:
#                 tfidf_filter [stu_key] = (rank, stu_val) #筛选构造带有排名的排序后的分词字典tfidf_filter
#
#         for k  in range(len(result[i])):
#             if result[i][k] in tfidf_filter.keys():
#                 data[i].append(result[i][k])
#
#     for i in range(len(data)):
#         data[i]="".join(data[i])
#     return data

def newwords_extract(corpus):
    f = open(corpus, mode="r", encoding="utf-8")
    data = f.read()
    f.close
    word_info_list = HanLP.extractWords(data, 10000, True)  # 第2个参数为提取的词语数量，第3个参数为是否过滤词典中已有的词语
    # for word in word_info_list:
    #     print(word.text, end=",")

    return word_info_list


#  return word_info_list

def newwords_process():
    CustomDictionary = JClass("com.hankcs.hanlp.dictionary.CustomDictionary")
    fileHandler = open('data/newword_dict.txt', 'r', encoding='utf-8')
    while True:
        # Get next line from file
        line = fileHandler.readline()
        #  line = line.decode('utf-8')
        # If line is empty then end of file reached
        if not line:
            break
        #        print(line.strip())
        CustomDictionary.add(line.strip())
        # Close Close
    fileHandler.close()


def keyword_extraction(data, mode):
    all_words = [i for item in data for i in item]
    wordcount = {}
    for word in all_words:
        wordcount[word] = wordcount.get(word, 0) + 1
    b = sorted(wordcount.items(), key=lambda x: x[1], reverse=True)  # 每个词排序
    print(b[0][1])
    word_list = []
    if mode == 'headline':  # 提取100词
        for key in range(len(b)):
            word_list.append(b[key][0])
            if len(word_list) >= HEADLINE_LEN:
                break
    if mode == 'abstract':  # 提取500词
        for key in range(len(b)):
            word_list.append(b[key][0])
            if len(word_list) >= ABSTRACT_LEN:
                break
    return word_list


#  print(a)

def segmentation(data):  # 分词操作

    # 将新词添加进入词典
    result_list = []
    result_str = []

    for i in tqdm(range(int(len(data)))):
        if pd.isnull(data[i]) == False:
            text = HanLP.segment(data[i])
            #   dic = load_file('../../HANLP/Introduction-NLP-master/data/dictionnary/stopwords.txt')
            dic = load_file('data/cn_stopwords.txt')
            result = remove_stop_words(text, dic)
            result_list.append(result)

    for i in range(int(len(result_list))):  # 将list转为str
        try:
            result_str.append(' '.join(result_list[i]))
        except:
            print('运行失败')
    return result_list, result_str


def baoliuci(head_data, abstract_data, extract_head, abstract_head):
    a = len(head_data)
    b = len(abstract_data)

    head_list = [[] for i in range(a)]
    abstract_list = [[] for j in range(b)]

    for i in range(a):
        for j in range(len(head_data[i])):
            if head_data[i][j] in extract_head:
                head_list[i].append(head_data[i][j])

    for i in range(b):
        for j in range(len(abstract_data[i])):
            if abstract_data[i][j] in abstract_head:
                abstract_list[i].append(abstract_data[i][j])

    # head_str = []
    # abstract_str =[]
    # for i in range(len(abstract_list)): #将list转为str
    #     head_str.append(' '.join(head_list[i]))
    #     abstract_str.append(' '.join(abstract_list[i]))
    return head_list, abstract_list




if __name__ == "__main__":
    # d = {'2022/03/28': [1,2,3], '2022/03/29': [20,50,10],'2022/04/01': [2,4,11],'2022/04/05':[3,5,61]}
