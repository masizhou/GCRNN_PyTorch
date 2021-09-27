import pandas as pd
import math


def read_file(filename):
    f = pd.read_excel(filename)
    # print(f.iloc[:, [1, 3, 4]])
    station_number = f.iloc[:, [1]]  # 后面要加“[]”，不然就自动转换成了series类型，不能使用iloc函数了
    lat = f.iloc[:, [3]]
    long = f.iloc[:, [4]]
    return station_number, lat, long


# 由经纬度计算节点之间的距离
# 参考博客：https://blog.csdn.net/weixin_45901519/article/details/108351201
def jisuan_jiedian_juli(filename):
    station_number, lat, long = read_file(filename)
    save = []
    for i in range(len(station_number)):
        for j in range(i+1, len(station_number)):
            lat1 = jiaodu_zhuanhuan_hudu(lat.iloc[i, 0])
            lat2 = jiaodu_zhuanhuan_hudu(lat.iloc[j, 0])
            long1 = jiaodu_zhuanhuan_hudu(long.iloc[i, 0])
            long2 = jiaodu_zhuanhuan_hudu(long.iloc[j, 0])
            a = lat1 - lat2  # 纬度差的绝对值
            b = long1 - long2  # 经度差的绝对值
            ant = pow(math.sin(a/2), 2) + math.cos(lat1)*math.cos(lat2)*pow(math.sin(b/2), 2)  # 公式
            s = 2*6378.137*math.asin(pow(ant, 0.5))
            save.append([station_number.iloc[i, 0], station_number.iloc[j, 0], s])
    print(save)
    f = pd.DataFrame(save, columns=["from", "to", "cost"])
    f.to_csv(r"C:\Users\MSZ\Desktop\GCRNN_PyTorch\data\Gansu_province\station_graph\distances_gansu.csv", index=None)


# 将角度转换为弧度
def jiaodu_zhuanhuan_hudu(degree_minute):
    degree = degree_minute // 100  # 度
    minute = degree_minute % 100  # 分
    degree = degree + minute / 60  # 将分转换为度，并相加
    degree = float("%.2f" % degree)  # 保留两位小数

    rad = degree * math.pi / 180
    return rad


if __name__ == "__main__":
    jisuan_jiedian_juli(r"C:\Users\MSZ\Desktop\GCRNN_PyTorch\data\Gansu_province\station_graph\gansu_zhandian_chuliqian.xlsx")
    print("finish!")
