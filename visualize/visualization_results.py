import numpy as np
from matplotlib import pyplot as plt

def visualization_dcrnn_prediction(filename: str):
    f = np.load(filename)
    prediction = f["prediction"] # (12, 256, 74)
    truth = f["truth"] # (12, 256, 74)

    plt.Figure()
    plt.grid(True, linestyle="-.", linewidth=0.5)
    plt.plot(prediction[1, :, 1])
    plt.plot(truth[1, :, 1])

    plt.legend(["prediction", "target"], loc="upper right")

    plt.show()



if __name__ == "__main__":
    visualization_dcrnn_prediction("../data/dcrnn_heilongjiang_predictions.npz")