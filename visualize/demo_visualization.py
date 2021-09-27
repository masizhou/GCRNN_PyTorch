import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torchvision

torchvision.models.resnet.resnet50()

# 显示中文
font = {'family':'SimHei', 'weight':'bold', 'size':'12'}
plt.rc('font', **font)               # 步骤一（设置字体的更多属性）
plt.rc('axes', unicode_minus=False)  # 步骤二（解决坐标轴负数的负号显示问题）


def get_temperature_data(file_name):
    temperature_data = np.load(file_name)

    return temperature_data


if __name__ == "__main__":
    temperature_data = get_temperature_data("../data/Guangdong_province/Guangdong_temperature/Guangdong_temperature.npy")
    print(temperature_data.shape) # (86, 1296, 11)
    # print(temperature_data)

    plt.plot(temperature_data[85, :180,:1], label='station1', color="blue")
    plt.plot(temperature_data[3, :180,:1], label='station2', color="green")
    plt.plot(temperature_data[4, :180,:1], label='station3', color="orange")

    plt.xlabel("Time/h")
    plt.ylabel("Temperature/C")
    plt.xticks(range(0, 181, 24))  # 设置x轴
    plt.legend()

    # plt.show()
    plt.savefig("../figures/schematic_diagram_of_spatial_correlation_of_temperature_data.png", dpi=600)
    plt.savefig("../figures/schematic_diagram_of_spatial_correlation_of_temperature_data.svg")