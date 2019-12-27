# -*- coding: utf-8 -*-
# main process

import os
import pickle
import warnings

from pandas import concat, Series
import matplotlib.pyplot as plt

from data_io import DataLoader
from model import StockModel
from model_configs import *

TUNE = True  # whether tuning
SAVE_PLOTS = True  # whether to save the prediction plots


def main():
    data_dir = "./data/raw_price_train"  # data directory(contains all stocks' history price data and sentiment scores)
    fig_save_dir = "./figs_new"  # result plots' save directory
    io_handler = DataLoader()
    n_forecast_step = 7  # future forecasting days
    exclude_test_dates = ["2015-12-25", "2015-12-26", "2015-12-27", "2015-12-31"]  # excluded dates in the test set
    submission_save_path = "./submission_new.pkl"  # final prediction save directory
    submission = []
    sarimax_feats = ["Open", "High", "Low", "Close", "Volume"]  # Features to be predicted
    hzt_feats = sarimax_feats + ["senti_score"]  # Features used to build the horizontal model
    y_col = "Adj Close"
    param_save_dir = "./tuned_params"  # tuned parameters save directory

    warnings.filterwarnings("ignore")
    # traverse all stock datasets
    for i, data_train, senti_score_train, senti_score_test in io_handler.load_datasets(data_dir, exclude_test_dates=exclude_test_dates):
        param_fp = os.path.join(param_save_dir, "params-stock_%d.json" % i)
        sarimax_train_data = data_train[sarimax_feats]
        stock_model = StockModel(sarimax_param_grid, hzt_param_grid, hzt_init_model)

        if TUNE:
            # tune all feature predictors
            stock_model.tune_sarimax(sarimax_train_data, scoring=sarimax_tune_scoring, train_size=train_size,
                                     refit=refit)
        else:
            tuned_params = stock_model.load_tuned_params(param_fp)
            stock_model.train_sarimax(sarimax_train_data, extern_feats_params=tuned_params["sarimax"])
        # predict the features
        sarimax_feats_pred = stock_model.sarimax_predict(n_forecast_step, data_train, index=senti_score_test.index)

        # get the training and test sets for the horizontal model
        hzt_train = concat((data_train, senti_score_train), axis=1, join="inner")
        hzt_x_train = hzt_train[hzt_feats]
        hzt_x_test = concat((sarimax_feats_pred, senti_score_test), axis=1, join="inner")
        hzt_x = concat((hzt_x_train, hzt_x_test), axis=0)
        m_train = hzt_x_train.shape[0]
        hzt_y_train = hzt_train[y_col].values
        hzt_x = scaler.fit_transform(hzt_x.values)
        hzt_x_train, hzt_x_test = hzt_x[:m_train, :], hzt_x[m_train:, :]

        if TUNE:
            # tune the horizontal model
            stock_model.tune_hzt(hzt_x_train, hzt_y_train, cv=hzt_cv, scoring=hzt_tune_scoring, n_jobs=n_jobs,
                                 refit=refit)
            stock_model.save_params(param_fp)
        else:
            stock_model.train_hzt(hzt_x_train, hzt_y_train, extern_params=tuned_params["hzt"])
        pred = stock_model.hzt_predict(hzt_x_test)

        # record the predictions
        submission.append(pred.tolist())
        pred = Series(data=pred, name="Adj Close", index=senti_score_test.index)
        print("Final predictions:")
        print(pred)

        if SAVE_PLOTS:
            # visualize the predictions
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.plot(data_train[y_col])
            ax.plot(pred, linewidth=1.5)
            ax.legend(["history", "prediction"])
            ax.set_xlabel("Date")
            ax.set_ylabel("Adj Close")
            ax.set_title("Prediction plot of stock %d" % i)
            plt.savefig(os.path.join(fig_save_dir, "%d-pred.png" % i))

    # save the submissions to file
    with open(submission_save_path, "wb") as f:
        pickle.dump(submission, f)
    print("Submission saved to file: %s" % submission_save_path)


if __name__ == '__main__':
    main()
