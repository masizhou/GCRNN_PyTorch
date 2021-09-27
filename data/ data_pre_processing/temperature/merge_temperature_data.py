import sys
sys.path.append("..")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_table_file(filename): # 读取.txt文件
    f = pd.read_table(filename, sep=" ", index_col=False)  # index_col=False则不会让第一列为index
    return f


def read_csv_file(filename): # 读取.csv文件
    f = pd.read_csv(filename, index_col=False)  # index_col=False则不会让第一列为index
    return f


def merge_data(f1, f2):  # 合并数据
    """
    :param f1:
    :param f2:
    :return:
    """
    # print(f2)
    f = pd.DataFrame(columns=f1.columns)  # 创建空的给定列名的DataFrame
    i = 0
    j = 0
    while i < len(f1): # 循环，按指定行数进行合并，比如读文件1的120行，再读取文件2的120行，等等
        i += 1152 # 24*day
        j += 144
        f = f.append(f1.iloc[i-1152:i, :])
        f = f.append(f2.iloc[j-144:j, :])

    print(f.shape)
    f.to_csv(r"C:\Users\MSZ\Desktop\GCRNN_PyTorch\data\Heilongjiang_province\Heilongjiang_temperature\heilongjiang_11_1_00-12_25_23.csv", index=None) # 存入指定文件




if __name__ == "__main__":
    # f1 = read_table_file(r"C:\Users\MSZ\Desktop\dataset\Heilongjiang\11-1-0~11-5-23\S202011061515065342100.txt")  # 11-1-0到11-5-23
    # f2 = read_table_file(r"C:\Users\MSZ\Desktop\dataset\Heilongjiang\11-06-00~11-10-23\S202011111012195646900.txt")  # 11-6-0到11-10-23
    f1 = read_csv_file(r"C:\Users\MSZ\Desktop\GCRNN_PyTorch\data\Heilongjiang_province\Heilongjiang_temperature\heilongjiang_11_1_00-12_19_23.csv")
    f2 = read_table_file(r"C:\Users\MSZ\Desktop\dataset\Heilongjiang\12-20-00~12-25-23\S202012262145518934800.txt")

    merge_data(f1, f2)

    print(f1.shape)
    print(f2.shape)

    print("finish!")
