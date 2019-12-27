# -*- coding: utf-8 -*-

import os

from pandas import read_csv, Timestamp, DataFrame, DatetimeIndex, Series


class DataLoader:
    
    @staticmethod
    def load_datasets(data_dir: str, start_date="2014-01-02", exclude_test_dates=None) -> (DataFrame, Series):
        """
        load the raw price training datasets with sentiment scores stock by stock
        :param data_dir: the directory that contains all stocks' history price data and training/test sentiment scores
        :param start_date: the data after `start_date` will be used, while the data before it will be ignored
        :param exclude_test_dates: the sentiment scores of these dates will not be used since the these dates won't have predictions
        :yield: (stock number, training price data, training sentiment scores, test sentiment scores)
        """
        for i in range(1, 9):
            print("=================================== Loading stock No.%d's datasets... ===================================" % i)
            # infer file paths
            raw_price_train_fp = os.path.join(data_dir, "%s_r_price_train.csv" % i)
            senti_score_train_fp = os.path.join(data_dir, "%s_r_senti_score_train.csv" % i)
            senti_score_test_fp = os.path.join(data_dir, "%s_r_senti_score_test.csv" % i)
            # read data files(history price, training sentiment scores and test sentiment scores)
            data_train = read_csv(raw_price_train_fp, header=0, index_col=0, parse_dates=["Date"])
            senti_score_train = read_csv(senti_score_train_fp, header=0, index_col=0, parse_dates=["Date"])
            senti_score_test = read_csv(senti_score_test_fp, header=0, index_col=0, parse_dates=["Date"])
            if exclude_test_dates is not None:
                exclude_test_dates = DatetimeIndex(exclude_test_dates)
                senti_score_test.drop(exclude_test_dates, axis=0, inplace=True)  # drop the dates which won't have predictions
            start_ts = Timestamp(start_date)
            # reserve the data after `start_date`
            senti_score_train = senti_score_train.loc[senti_score_train.index >= start_ts]
            data_train = data_train.loc[data_train.index >= start_ts, :]
            yield i, data_train, senti_score_train, senti_score_test

