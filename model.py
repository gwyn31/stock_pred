# -*- coding: utf-8 -*-
# model construction

from time import time
import json
from itertools import product

from numpy import ndarray, array, mean, sqrt, nan, concatenate, cumsum, abs as np_abs
from pandas import Series, DataFrame
from sklearn.base import RegressorMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from statsmodels.tsa.api import SARIMAX


def mape(y_true: ndarray, y_pred: ndarray) -> float:
    """
    Mean Absolute Percentage Error(MAPE) metric
    :param y_true: ground truth array
    :param y_pred: prediction array
    :return: MAPE value
    """
    y1, y2 = y_true.ravel(), y_pred.ravel()
    return float(mean(np_abs((y1 - y2) / y1)))


def rmse(y_true: ndarray, y_pred: ndarray) -> float:
    """
    RMSE metric
    :param y_true: ground truth array
    :param y_pred: prediction array
    :return: RMSE value
    """
    return sqrt(mean_squared_error(y_true, y_pred))


scoring_alias = {
    "mape": mape, "rmse": rmse, "mae": mean_absolute_error
}
scorer_alias = {
    "mape": make_scorer(mape, greater_is_better=False),
    "rmse": make_scorer(rmse, greater_is_better=False),
    "mae": "neg_mean_absolute_error"
}


class SARIMAX111:

    """
    SARIMAX model with (p, d, q) = (1, 1, 1)(because of the descriptive analysis)
    """

    def __init__(self, seasonal_order=(1, 1, 1, 5), trend="c"):
        """
        :param seasonal_order: seasonal_order: seasonal parameter: (P, D, Q, s)
        :param trend: the trend parameter
        """
        self.seasonal_order = seasonal_order
        self.trend = trend
        self._fitted = None

    def get_fitted(self):
        """
        retrieve the fitted model
        :return: the fitted model
        """
        return self._fitted

    def train(self, diff1_seq: Series):
        """
        train the SARIMAX model with a given 1-order difference sequence
        :param diff1_seq: a 1-order difference time series
        :return: self
        """
        sarimax = SARIMAX(diff1_seq, order=(1, 1, 1), seasonal_order=self.seasonal_order, trend=self.trend)
        self._fitted = sarimax.fit(disp=-1)
        return self

    def predict(self, steps: int, original_seq: Series) -> ndarray:
        """
        make predictions(given the number of expected predictions)
        :param steps: number of expected predictions
        :param original_seq: original time series(before 1-order difference)
        :return: prediction array
        """
        if self._fitted is None:
            raise AttributeError("Model not fitted.")
        else:
            pred = self._fitted.forecast(steps=steps)
            val = original_seq.values[-1]
            actual_pred = concatenate(([val], array(pred)))
            return cumsum(actual_pred)[1:]


class SARIMAX111Tuner:

    """
    Class for tuning the SARIMAX(p=1, d=1, q=1) model
    """

    def __init__(self, param_grid: dict, scoring="mape", train_size=0.85, refit=True):
        """
        :param param_grid: all parameter combinations, keys are 'seasonal_order' and 'trend'
        :param scoring: name of scoring criterion("mape", "rmse" or "mae")
        :param train_size: the percent of training set
        :param refit: whether to refit the model using all data
        """
        self.param_grid = param_grid
        self.scoring = scoring
        self.train_size = train_size
        self.refit = refit
        self._scorer = scoring_alias.get(scoring)

    def grid_search(self, seq: Series) -> dict:
        """
        Perform grid search for parameters: `seasonal_order`(P, D, Q, s) and `trend`
        :param seq: training series, it will be automatically made an 1-order difference for training the SARIMAX model
        :return: dict of best parameter and best model
        """
        param_list = list(product(self.param_grid.get("seasonal_order"), self.param_grid.get("trend")))
        tune_scores = []
        diff1_seq = seq.diff(1).dropna()  # get its 1-order difference sequence
        m = len(seq)
        m_train = int(m * self.train_size)
        m_test = m - m_train
        for so, trend in param_list:  # traverse all parameter combinations
            sarimax = SARIMAX111(seasonal_order=so, trend=trend)
            # split training and test sets
            seq_train = diff1_seq.iloc[:m_train]
            seq_test = seq.iloc[m_train:]
            try:
                sarimax.train(seq_train)  # train the SARIMAX model
                pred = sarimax.predict(m_test, seq.iloc[:m_train])  # make predictions on the test set
                score = self._scorer(seq_test.values, pred)  # evaluate the predictions
            except Exception as e:
                score = nan
            tune_scores.append(score)  # record the test score
        best_idx = Series(tune_scores).dropna().argmin()  # find the best parameters
        print("Tuning best score: %g" % tune_scores[best_idx])
        best_param = {"seasonal_order": param_list[best_idx][0], "trend": param_list[best_idx][1]}
        best_model = None
        if self.refit:  # if refit, using the whole series to train the SARIMAX model
            sarimax = SARIMAX111(**best_param)
            sarimax.train(diff1_seq)
            best_model = sarimax
        return {"best_param": best_param, "best_model": best_model}


class StockModel:

    """
    Stock predictor
    """

    def __init__(self,
                 sarimax_param_grid: dict,
                 hzt_mdl_param_grid: dict,
                 hzt_init_model: RegressorMixin):
        """
        :param sarimax_param_grid: SARIMAX(p=1, d=1, q=1)'s tuning parameter grid
        :param hzt_mdl_param_grid: Horizontal model's tuning parameter grid
        :param hzt_init_model: Initial horizontal model
        """
        self.sarimax_param_grid = sarimax_param_grid
        self.hzt_mdl_param_grid = hzt_mdl_param_grid
        self._hzt_init_model = hzt_init_model
        self.tuned_sarimax_params = None  # if set, will be: {feature name: dict of tuned parameters}
        self.tuned_hzt_params = None  # if set, will be the tuned horizontal model parameters
        self.sarimaxs = None  # trained SARIMAX models: {feature name: trained SARIMAX111}
        self.hzt = None  # trained horizontal model

    def load_tuned_params(self, fp: str, encoding="utf-8") -> (dict, dict):
        """
        Load the tuned parameters
        :param fp: tuned parameter file(.json)
        :param encoding: file's encoding
        :return: tuned SARIMAX parameters, tuned horizontal model's parameters
        """
        with open(fp, "r", encoding=encoding) as f:
            params = json.load(f)
        print("Tuned parameters loaded from: %s" % fp)
        return params["sarimax"], params["hzt"]

    def save_params(self, fp: str, encoding="utf-8"):
        """
        Save the tuned parameters
        :param fp: target file path
        :param encoding: file's encoding
        """
        all_params = {"sarimax": self.tuned_sarimax_params, "hzt": self.tuned_hzt_params}
        with open(fp, "w", encoding=encoding) as f:
            json.dump(all_params, f)
        print("Tuned parameters saved to: %s" % fp)

    def tune_sarimax(self, data: DataFrame, scoring="mape", train_size=0.85, refit=True):
        """
        Tune each feature's SARIMAX model
        :param data: Stock history price dataset
        :param scoring: tuning scoring criterion
        :param train_size: training set size(in percentage)
        :param refit: whether to refit the model using the whole series
        :return: self
        """
        tuner = SARIMAX111Tuner(self.sarimax_param_grid, scoring=scoring, train_size=train_size, refit=refit)
        tuned_sarimax_params = dict()
        tuned_sarimax = dict()
        for col in data.columns:  # traverse all feature sequences
            print("Tuning feature: %s's SARIMAX model..." % col)
            seq = data[col]
            tic = time()
            result = tuner.grid_search(seq)  # grid search
            toc = time()
            print("Done, takes %g secs." % (toc - tic))
            tuned_sarimax_params[col] = result.get("best_param")
            if refit:
                tuned_sarimax[col] = result.get("best_model")
            self.tuned_sarimax_params = tuned_sarimax_params
            self.sarimaxs = tuned_sarimax
        return self

    def tune_hzt(self, x: ndarray, y: ndarray, cv=3, scoring="mape", n_jobs=2, refit=True):
        """
        Tune the horizontal predictor
        :param x: feature data
        :param y: ground truth
        :param cv: number of cross validation folds
        :param scoring: tuning scoring criterion
        :param n_jobs: number of parallel CPUs
        :param refit: whether to refit the model using all data
        :return: self
        """
        scoring = scorer_alias.get(scoring)
        model = self._hzt_init_model
        print("Tuning horizontal model: %s..." % model.__class__.__name__)
        tuner = GridSearchCV(model, self.hzt_mdl_param_grid,
                             scoring=scoring, cv=cv, n_jobs=n_jobs, refit=refit)
        tic = time()
        tuner.fit(x, y)
        toc = time()
        print("Done, tuning best score: %g, takes %g secs." % (abs(tuner.best_score_), toc - tic))
        self.tuned_hzt_params = tuner.best_params_
        if refit:
            self.hzt = tuner.best_estimator_
        return self

    def _check_extern_or_tuned_params(self, attr_name: str, extern_params=None, model_name="horizontal model") -> dict:
        if extern_params is not None:
            print("Training %s using external parameters..." % model_name)
            return extern_params
        elif getattr(self, attr_name) is not None:
            print("Training %s using tuned parameters..." % model_name)
            return self.tuned_sarimax_params
        else:
            raise ValueError("None of external or tuned parameters provided.")

    def train_sarimax(self, data: DataFrame, extern_feats_params=None):
        """
        train feature sequence ARIMA predictors using tuned parameters or external parameters(prior)
        :param data: Stock history price dataset
        :param extern_feats_params: external SARIMAX parameters
        :return: self
        """
        col_params = self._check_extern_or_tuned_params("tuned_arima_params", extern_params=extern_feats_params,
                                                        model_name="feature sequence ARIMA predictors")
        trained_sarimaxs = dict()
        for col in data.columns:
            seq = data[col]
            diff1_seq = seq.diff(1).dropna()
            sarimax = SARIMAX111(**col_params[col])
            sarimax.train(diff1_seq)
            trained_sarimaxs[col] = sarimax
        print("All done.")
        self.sarimaxs = trained_sarimaxs
        return self

    def train_hzt(self, x: ndarray, y: ndarray, extern_params=None):
        """
        train horizontal model using tuned parameters or external parameters(prior)
        :param x: feature data
        :param y: ground truth
        :param extern_params: external parameters
        :return: self
        """
        params = self._check_extern_or_tuned_params("tuned_hzt_params", extern_params=extern_params,
                                                    model_name="horizontal model")
        model = object.__new__(self._hzt_init_model.__class__)
        model.__init__(**params)
        model.fit(x, y)
        print("Done.")
        self.hzt = model
        return self

    def sarimax_predict(self, steps: int, original_seqs: DataFrame, index=None) -> DataFrame:
        """
        Get features' future predictions
        :param steps: number of predictions
        :param original_seqs: original feature sequences
        :param index: future index
        :return: prediction dataframe
        """
        if self.sarimaxs is None:
            raise AttributeError("SARIMAX models not fitted")
        else:
            feat_preds = DataFrame()
            for col, sarimax in self.sarimaxs.items():
                pred = sarimax.predict(steps, original_seqs[col])
                feat_preds[col] = pred
            if index is not None:
                feat_preds.index = index
            return feat_preds

    def hzt_predict(self, x: ndarray) -> ndarray:
        """
        Horizontal model's prediction
        :param x: feature data
        :return: prediction array
        """
        if self.sarimaxs is None:
            raise AttributeError("Horizontal model not fitted")
        else:
            return self.hzt.predict(x)

