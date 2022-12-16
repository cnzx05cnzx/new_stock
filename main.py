# coding=gbk
import pandas as pd
from pandas import read_parquet


def fun(x):
    s = str(x[0])
    for i in range(1, len(x)):
        s += ',' + str(x[i])
    return s
res1=[1,2,3,4,5,6,7]
res2=[4,6,9]
print(fun(res2))

