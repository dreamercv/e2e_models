# -*- encoding: utf-8 -*-
'''
@File         :train.py
@Date         :2026/03/07 13:31:48
@Author       :Binge.Van
@E-mail       :1367955240@qq.com
@Version      :V1.0.0
@Description  :

'''

from config.config import configs
from dataset.dataset import Dataset

def main():
    dataset = Dataset(configs)
    print(dataset.__len__())
    dataset.__getitem__(0)
    # for i in range(dataset.__len__()):
    #     dataset.__getitem__(i)


