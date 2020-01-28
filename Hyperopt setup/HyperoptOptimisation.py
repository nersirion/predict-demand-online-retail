import numpy as np
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
import xgboost as xgb
import catboost as ctb
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials, hp

# XGB parameters
xgb_reg_params = {
    'learning_rate':    hp.uniform('learning_rate',    0.005, 0.31, ),
    'max_depth':        hp.choice('max_depth',        np.arange(5, 16, 1, dtype=int)),
    'boosting_type':     'gbdt',
    'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 0.95),
    'subsample':        hp.uniform('subsample', 0.1, 1),
    'n_estimators':     1000,
    'gamma':            hp.uniform('gamma',     0.01, 0.95),
    'objective': 'reg:squarederror',
    'num_leaves':       hp.choice('num_leaves',        np.arange(600, 1200, 24, dtype=int)),               # 1024,440    # 2^max_depth < num_leaves ?
    'min_gain_to_split': hp.uniform('min_gain_to_split', 0.01, 0.25),
    'min_child_weight':  hp.uniform('min_child_weight', 0.1, 3),
    'lambda_l1':        hp.uniform('l1', 0, 3),
    'lambda_l2':        hp.uniform('l2', 0, 3),
}
xgb_fit_params = {
    'eval_metric': 'rmse',
    'early_stopping_rounds': 100,
    'verbose': False,
    
}
xgb_para = dict()
xgb_para['reg_params'] = xgb_reg_params
xgb_para['fit_params'] = xgb_fit_params
xgb_para['loss_func' ] = lambda y, pred: np.sqrt(mean_squared_error(y, pred))


# LightGBM parameters
lgb_reg_params = {
    'learning_rate':    hp.uniform('learning_rate',    0.005, 0.31, ),
    'max_depth':        hp.choice('max_depth',        np.arange(5, 20, 1, dtype=int)),
    'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
    'colsample_bytree': hp.uniform('colsample_bytree',0.1, 0.9),
    'subsample':        hp.uniform('subsample', 0.1, 1),
    'n_estimators':     1000,
    'boosting_type':      'gbdt', 
    'objective': 'regression',
    'metric': 'rmse',
    'feature_fraction':  hp.uniform('feature_fraction', 0.1, 0.99),       
    'bagging_fraction':  hp.uniform('bagging_fraction', 0.1, 0.95),       
    'gamma':            hp.uniform('gamma',     0.01, 0.5),
  
    'num_leaves':       hp.choice('num_leaves',        np.arange(900, 1200, 24, dtype=int)),              
    'min_gain_to_split': hp.uniform('min_gain_to_split', 0.01, 0.25),
    'min_child_weight':  hp.uniform('min_child_weight', 0.1, 3),
    'lambda_l1':        hp.uniform('l1', 1, 3),
    'lambda_l2':        hp.uniform('l2', 1, 3),
}
lgb_fit_params = {
    #'eval_metric': 'l2',
    #'verbose_eval ': 50,
    'early_stopping_rounds': 100,
    'verbose': False,

   
}
lgb_para = dict()
lgb_para['reg_params'] = lgb_reg_params
lgb_para['fit_params'] = lgb_fit_params
lgb_para['loss_func' ] = lambda y, pred: np.sqrt(mean_squared_error(y, pred))


# CatBoost parameters
ctb_reg_params = {
    'learning_rate':     hp.choice('learning_rate',     np.arange(0.05, 0.31, 0.05)),
    'max_depth':         hp.choice('max_depth',         np.arange(5, 16, 1, dtype=int)),
    'colsample_bylevel': hp.choice('colsample_bylevel', np.arange(0.3, 0.8, 0.1)),
    'n_estimators':      1000,
    'eval_metric':       'RMSE',
}
ctb_fit_params = {
    'early_stopping_rounds': 100,
    'verbose': False
}
ctb_para = dict()
ctb_para['reg_params'] = ctb_reg_params
ctb_para['fit_params'] = ctb_fit_params
ctb_para['loss_func' ] = lambda y, pred: np.sqrt(mean_squared_error(y, pred))





class HPOpt(object):

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test  = x_test
        self.y_train = y_train
        self.y_test  = y_test

    def process(self, fn_name, space, trials, algo, max_evals):
        fn = getattr(self, fn_name)
        try:
            result = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return result, trials

    def xgb_reg(self, para):
        reg = xgb.XGBRegressor(**para['reg_params'])
        return self.train_reg(reg, para)

    def lgb_reg(self, para):
        reg = lgb.LGBMRegressor(**para['reg_params'])
        return self.train_reg(reg, para)

    def ctb_reg(self, para):
        reg = ctb.CatBoostRegressor(**para['reg_params'])
        return self.train_reg(reg, para)

    def train_reg(self, reg, para):
        reg.fit(self.x_train, self.y_train,
                eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)],
                **para['fit_params'])
        pred = reg.predict(self.x_test)
        loss = para['loss_func'](self.y_test, pred)
        return {'loss': loss, 'status': STATUS_OK}