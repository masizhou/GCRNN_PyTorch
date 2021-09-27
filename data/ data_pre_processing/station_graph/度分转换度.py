import pandas as pd
import math


def read_file(filename):
    f = pd.read_excel(filename)
    print(f.iloc[:, [1, 3, 4]])
    station_number = f.iloc[:, [1]]  # 后面要加“[]”，不然就自动转换成了series类型，不能使用iloc函数了
    lat = f.iloc[:, [3]] # 纬度
    long = f.iloc[:, [4]] # 经度
    return station_number, lat, long


def dufen_zhuanhuan_du(degree_minute):  # 由于源文件是将度和分放一起了，所以这个函数作用是要分离开
    degree = degree_minute // 100   # 度
    minute = degree_minute % 100  # 分
    degree = degree + minute / 60  # 将分转换为度，并相加
    degree = "%.2f" % degree  # 保留两位小数
    return degree


if __name__ == "__main__":
    station_number, lat, long = read_file(r"C:\Users\MSZ\Desktop\GCRNN_PyTorch\data\Heilongjiang_province\station_graph\heilongjiang_zhandian_chuliqian.xlsx") # 站点、纬度、经度
    print(type(lat))
    save = [] # 保存计算的值
    for i in range(len(station_number)):
        lat1 = dufen_zhuanhuan_du(lat.iloc[i, 0])  # 纬度
        print()
        long1 = dufen_zhuanhuan_du(long.iloc[i, 0])  # 经度
        save.append([lat1, long1])

    print(save)
    f = pd.DataFrame(save, columns=["lat", "long"])
    f.to_csv(r"C:\Users\MSZ\Desktop\GCRNN_PyTorch\data\Heilongjiang_province\station_graph\latitude_and_longitude.csv", index=None)

    print("finish!")
