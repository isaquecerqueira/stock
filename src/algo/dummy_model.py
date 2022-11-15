import logging
import pandas as pd
import random
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklego.preprocessing import RepeatingBasisFunction


def create_features(df_stock, nlags=10):
    def tagger(row):
        if row['next'] < row['lag_0']:
            return 'Sell'
        else:
            return 'Buy'

    columns = [f'lag_{lag}' for lag in reversed(range(0, nlags))]
    columns += ['out']

    # lags features
    for lag in range(0, nlags):
        df_stock[f'lag_{lag}'] = df_stock['close'].shift(lag)
    df_stock['next'] = df_stock['close'].shift(-1)
    df_stock['out'] = df_stock.apply(tagger, axis=1)
    df_clean = df_stock[columns].dropna(axis=0)

    # moving average features
    ma_day = [10, 20, 50]
    for ma in ma_day:
        column_name = f"MA_{ma}"
        df_clean[column_name] = df_clean['lag_0'].rolling(ma).mean()

    # Time features
    df_time = pd.DataFrame(index=df_clean.index)
    df_time['day_of_year'] = pd.to_datetime(df_clean.index)
    rbf = RepeatingBasisFunction(n_periods=12,
                              column= 'day_of_year',
                              remainder="drop")
    rbf.fit(df_time)
    df_time = pd.DataFrame(index=df_clean.index,
                    data=rbf.transform(df_time))  
    df_clean = pd.merge(df_clean, df_time, left_index=True, right_index=True)

    return df_clean

def create_X_Y(df_lags):
    df_lags = df_lags.iloc[:-1 , :]
    X = df_lags.drop('out', axis=1)
    Y = df_lags[['out']]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, Y


class Stock_model(BaseEstimator, TransformerMixin):

    def __init__(self, data_fetcher):
        self.log = logging.getLogger()
        self.lg = RandomForestClassifier(max_depth=5, random_state=0)
        self._data_fetcher = data_fetcher
        self.log.warning('here')

    def fit(self, X, Y=None):
        data = self._data_fetcher(X)
        df_features = create_features(data)
        df_features, Y = create_X_Y(df_features)
        self.lg.fit(df_features, Y)
        return self

    def predict(self, X, Y=None):
        print(X)
        data = self._data_fetcher(X, last=True)
        print(data)
        df_features = create_features(data)
        print(df_features)
        df_features, Y = create_X_Y(df_features)
        predictions = self.lg.predict(df_features)

        return predictions.flatten()[-1]
