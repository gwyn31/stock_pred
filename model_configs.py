# -*- coding: utf-8 -*-

##### configurations

### SARIMAX model configuration
# parameter grid(all seasonal 'PDQs' with 'trend' combinations) of tuning
sarimax_params = {
    "P": [0, 1, 2], "D": [0, 1, 2], "Q": [0, 1, 2], "s": [0, 5, 14],
    "trend": ["c", "t", "ct"]
}
# scoring criterion of tuning
sarimax_tune_scoring = "mape"
# training set size(in percentage) of the train/validation split during SARIMAX tuning
train_size = 0.88

### Horizontal model configuration
# model type
hzt_model_type = "svr"
# model initialization parameters
hzt_model_init_params = {}
# parameter grid of tuning
hzt_param_grid = {
    "C": [0.1, 0.3, 1.0, 3.0, 10.0],
    "epsilon": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
    "gamma": [0.01, 0.03, 0.1, 0.3, 1.0]
}
# scoring criterion of tuning
hzt_tune_scoring = "mape"
# cross validation folds
hzt_cv = 3
# feature scaling method
scaling = "minmax"
# feature scaling parameters
scaling_param = {}

### Other configuration
# number of jobs for parallel tuning
n_jobs = 2
# whether to refit the model after tuning(using all training data)
refit = True


##### below are some pre-processing of the configurations given above
from itertools import product

from sklearn.linear_model import Lasso, Ridge, Huber
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


hzt_model_alias = {
    "lasso": Lasso, "ridge": Ridge, "huber": Huber, "svr": SVR, 
    "dt": DecisionTreeRegressor, 
    "gbdt": GradientBoostingRegressor, "rf": RandomForestRegressor
}
scaling_alias = {
    "minmax": MinMaxScaler, "zscore": StandardScaler, "robust_zscore": RobustScaler
}
sarimax_param_grid = {
    "seasonal_order": [tup for tup in product(sarimax_params["P"], sarimax_params["D"], sarimax_params["Q"], sarimax_params["s"])],
    "trend": sarimax_params["trend"]
}
hzt_init_model = hzt_model_alias.get(hzt_model_type)(**hzt_model_init_params)
scaler = scaling_alias.get(scaling)(**scaling_param)
