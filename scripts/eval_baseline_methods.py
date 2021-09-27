import argparse
import numpy as np
import pandas as pd
import yaml

from statsmodels.tsa.vector_ar.var_model import VAR
from lib import utils
from lib.metrics import masked_rmse_np, masked_mape_np, masked_mae_np
from lib.utils import StandardScaler
from lib.utils import load_graph_data



def historical_average_predict(temperature_data, period=24 * 7, test_ratio=0.2, null_val=0.):
    """
    Calculates the historical average of sensor reading.
    :param data:
    :param period: default 1 week.
    :param test_ratio:
    :param null_val: default 0.
    :return:
    """
    data = temperature_data.transpose(1, 0, 2)[:, :, :1]
    num_samples, num_nodes, features = data.shape
    n_test = int(round(num_samples * test_ratio))
    n_train = num_samples - n_test
    y_test = data[-n_test:, ...]
    y_predict = np.copy(y_test)

    for i in range(n_train, min(num_samples, n_train + period)):
        inds = [j for j in range(i % period, n_train, period)]
        historical = data[inds, ...]
        y_predict[i - n_train, ...] = historical[historical != null_val].mean()
    # Copy each period.
    for i in range(n_train + period, num_samples, period):
        size = min(period, num_samples - i)
        start = i - n_train
        y_predict[start:start + size, ...] = y_predict[start - period: start + size - period, ...]
    return y_predict, y_test


def static_predict(temperature_data, n_forward, test_ratio=0.2):
    """
    Assumes $x^{t+1} = x^{t}$
    :param data:
    :param n_forward:
    :param test_ratio:
    :return:
    """
    data = temperature_data.transpose(1, 0, 2)[:, :, :1]
    num_samples, num_nodes, features = data.shape
    test_num = int(round(num_samples * test_ratio))
    y_test = data[-test_num:, ...]
    y_predict = data[num_samples-test_num-n_forward:num_samples-n_forward, ...] # 把从前n_forward起的值，作为预测值
    return y_predict, y_test


def var_predict(temperature_data, n_forwards=(1, 3), n_lags=4, test_ratio=0.2):
    """
    Multivariate time series forecasting using Vector Auto-Regressive Model.
    :param df: pandas.DataFrame, index: time, columns: sensor id, content: data.
    :param n_forwards: a tuple of horizons.
    :param n_lags: the order of the VAR model.
    :param test_ratio:
    :return: [list of prediction in different horizon], dt_test
    """
    data = temperature_data.transpose(1, 0, 2)[:, :, 0]
    num_samples, num_output = data.shape
    n_test = int(round(num_samples * test_ratio))
    n_train = num_samples - n_test
    train_data, test_data = data[:n_train, ...], data[n_train:, ...]

    scaler = StandardScaler(mean=train_data.mean(), std=train_data.std())
    _data = scaler.transform(train_data)
    var_model = VAR(_data)
    var_result = var_model.fit(n_lags)
    max_n_forwards = np.max(n_forwards)
    # Do forecasting.
    result = np.zeros(shape=(len(n_forwards), n_test, num_output))
    start = n_train - n_lags - max_n_forwards + 1
    for input_ind in range(start, num_samples - n_lags):
        prediction = var_result.forecast(scaler.transform(data[input_ind: input_ind + n_lags]), max_n_forwards)
        for i, n_forward in enumerate(n_forwards):
            result_ind = input_ind - n_train + n_lags + n_forward - 1
            if 0 <= result_ind < n_test:
                result[i, result_ind, :] = prediction[n_forward - 1, :]

    predicts = []
    for i, n_forward in enumerate(n_forwards):
        predict = scaler.inverse_transform(result[i])
        predicts.append(predict)
    return predicts, test_data


def dcrnn_predict(prediction_data, horizon):
    data = np.load(prediction_data)
    prediction = data["prediction"][:horizon, ...]
    truth = data["truth"][:horizon, ...]

    return prediction, truth


def chebcrnn_predict(prediction_data, horizon):
    data = np.load(prediction_data)
    prediction = data["prediction"][:horizon, ...]
    truth = data["truth"][:horizon, ...]

    return prediction, truth


def fcrnn_predict(prediction_data, horizon):
    data = np.load(prediction_data)
    prediction = data["prediction"][:horizon, ...]
    truth = data["truth"][:horizon, ...]

    return prediction, truth


def gcrnn_predict(prediction_data, horizon):
    data = np.load(prediction_data)
    prediction = data["prediction"][:horizon, ...]
    truth = data["truth"][:horizon, ...]

    return prediction, truth


def eval_static(temperature_data):
    logger.info('Static')
    horizons = [1, 3, 6, 12]
    logger.info('\t'.join(['Model', 'Horizon', 'RMSE', 'MAPE', 'MAE']))
    for horizon in horizons:
        y_predict, y_test = static_predict(temperature_data, n_forward=horizon, test_ratio=0.2)
        rmse = masked_rmse_np(preds=y_predict, labels=y_test, null_val=0)
        mape = masked_mape_np(preds=y_predict, labels=y_test, null_val=0)
        mae = masked_mae_np(preds=y_predict, labels=y_test, null_val=0)
        line = 'Static\t%d\t%.2f\t%.2f\t%.2f' % (horizon, rmse, mape * 100, mae)
        logger.info(line)


def eval_historical_average(temperature_data, period):
    y_predict, y_test = historical_average_predict(temperature_data, period=period, test_ratio=0.2)
    rmse = masked_rmse_np(preds=y_predict, labels=y_test, null_val=0)
    mape = masked_mape_np(preds=y_predict, labels=y_test, null_val=0)
    mae = masked_mae_np(preds=y_predict, labels=y_test, null_val=0)
    logger.info('Historical Average')
    logger.info('\t'.join(['Model', 'Horizon', 'RMSE', 'MAPE', 'MAE']))
    for horizon in [1, 3, 6, 12]:
        line = 'HA\t%d\t%.2f\t%.2f\t%.2f' % (horizon, rmse, mape * 100, mae)
        logger.info(line)


def eval_var(temperature_data, n_lags=3):
    n_forwards = [1, 3, 6, 12]
    y_predicts, y_test = var_predict(temperature_data, n_forwards=n_forwards, n_lags=n_lags,
                                     test_ratio=0.2)
    logger.info('VAR (lag=%d)' % n_lags)
    logger.info('Model\tHorizon\tRMSE\tMAPE\tMAE')
    for i, horizon in enumerate(n_forwards):
        rmse = masked_rmse_np(preds=y_predicts[i], labels=y_test, null_val=0)
        mape = masked_mape_np(preds=y_predicts[i], labels=y_test, null_val=0)
        mae = masked_mae_np(preds=y_predicts[i], labels=y_test, null_val=0)
        line = 'VAR\t%d\t%.2f\t%.2f\t%.2f' % (horizon, rmse, mape * 100, mae)
        logger.info(line)


def eval_dcrnn(prediction_data):
    logger.info('DCRNN')
    horizons = [1, 3, 6, 12]
    logger.info('\t'.join(['Model', 'Horizon', 'RMSE', 'MAPE', 'MAE']))
    for horizon in horizons:
        y_predict, y_test = dcrnn_predict(prediction_data, horizon)
        rmse = masked_rmse_np(preds=y_predict, labels=y_test, null_val=0)
        mape = masked_mape_np(preds=y_predict, labels=y_test, null_val=0)
        mae = masked_mae_np(preds=y_predict, labels=y_test, null_val=0)
        line = 'DCRNN\t%d\t%.2f\t%.2f\t%.2f' % (horizon, rmse, mape * 100, mae)
        logger.info(line)


def eval_chebcrnn(prediction_data):
    logger.info('ChebCRNN')
    horizons = [1, 3, 6, 12]
    logger.info('\t'.join(['Model', 'Horizon', 'RMSE', 'MAPE', 'MAE']))
    for horizon in horizons:
        y_predict, y_test = chebcrnn_predict(prediction_data, horizon)
        rmse = masked_rmse_np(preds=y_predict, labels=y_test, null_val=0)
        mape = masked_mape_np(preds=y_predict, labels=y_test, null_val=0)
        mae = masked_mae_np(preds=y_predict, labels=y_test, null_val=0)
        line = 'ChebCRNN\t%d\t%.2f\t%.2f\t%.2f' % (horizon, rmse, mape * 100, mae)
        logger.info(line)


def eval_fcrnn(prediction_data):
    logger.info('FCRNN')
    horizons = [1, 3, 6, 12]
    logger.info('\t'.join(['Model', 'Horizon', 'RMSE', 'MAPE', 'MAE']))
    for horizon in horizons:
        y_predict, y_test = gcrnn_predict(prediction_data, horizon)
        rmse = masked_rmse_np(preds=y_predict, labels=y_test, null_val=0)
        mape = masked_mape_np(preds=y_predict, labels=y_test, null_val=0)
        mae = masked_mae_np(preds=y_predict, labels=y_test, null_val=0)
        line = 'FCRNN\t%d\t%.2f\t%.2f\t%.2f' % (horizon, rmse, mape * 100, mae)
        logger.info(line)


def eval_gcrnn(prediction_data):
    logger.info('GCRNN')
    horizons = [1, 3, 6, 12]
    logger.info('\t'.join(['Model', 'Horizon', 'RMSE', 'MAPE', 'MAE']))
    for horizon in horizons:
        y_predict, y_test = gcrnn_predict(prediction_data, horizon)
        rmse = masked_rmse_np(preds=y_predict, labels=y_test, null_val=0)
        mape = masked_mape_np(preds=y_predict, labels=y_test, null_val=0)
        mae = masked_mae_np(preds=y_predict, labels=y_test, null_val=0)
        line = 'GCRNN\t%d\t%.2f\t%.2f\t%.2f' % (horizon, rmse, mape * 100, mae)
        logger.info(line)


def main(args):
    temperature_data = np.load(args.temperature_filename)

    eval_static(temperature_data)
    eval_historical_average(temperature_data, period=7 * 24)
    eval_var(temperature_data, n_lags=3)

    # logger.info("gansu:")
    # eval_dcrnn(r"C:\Users\MSZ\Desktop\GCRNN_PyTorch\data\dcrnn_gansu_predictions.npz")
    # eval_chebcrnn(r"C:\Users\MSZ\Desktop\GCRNN_PyTorch\data\chebcrnn_gansu_predictions.npz")
    # eval_fcrnn(r"C:\Users\MSZ\Desktop\GCRNN_PyTorch\data\fcrnn_gansu_predictions.npz")
    # eval_gcrnn(r"C:\Users\MSZ\Desktop\GCRNN_PyTorch\data\gcrnn_gansu_predictions.npz")
    #
    # logger.info("guangdong:")
    # eval_dcrnn(r"C:\Users\MSZ\Desktop\GCRNN_PyTorch\data\dcrnn_guangdong_predictions.npz")
    # eval_chebcrnn(r"C:\Users\MSZ\Desktop\GCRNN_PyTorch\data\chebcrnn_guangdong_predictions.npz")
    # eval_fcrnn(r"C:\Users\MSZ\Desktop\GCRNN_PyTorch\data\fcrnn_guangdong_predictions.npz")
    # eval_gcrnn(r"C:\Users\MSZ\Desktop\GCRNN_PyTorch\data\gcrnn_guangdong_predictions.npz")

    logger.info("heilongjiang:")
    eval_dcrnn(r"C:\Users\MSZ\Desktop\GCRNN_PyTorch\data\dcrnn_heilongjiang_predictions.npz")
    eval_chebcrnn(r"C:\Users\MSZ\Desktop\GCRNN_PyTorch\data\chebcrnn_heilongjiang_predictions.npz")
    eval_fcrnn(r"C:\Users\MSZ\Desktop\GCRNN_PyTorch\data\fcrnn_heilongjiang_predictions.npz")
    eval_gcrnn(r"C:\Users\MSZ\Desktop\GCRNN_PyTorch\data\gcrnn_heilongjiang_predictions.npz")


if __name__ == '__main__':
    logger = utils.get_logger('data/model', 'Baseline')
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperature_filename',
                        default="data/Heilongjiang_province/Heilongjiang_temperature/Heilongjiang_temperature.npy",
                        type=str,
                        help='Raw temperature readings.')
    args = parser.parse_args()
    main(args)
