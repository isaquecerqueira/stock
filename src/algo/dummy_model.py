import logging
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklego.preprocessing import RepeatingBasisFunction
#from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def create_features(df_stock, nlags=5):
    def tagger(row):
        if row['next'] < row['lag_0']:
            return 'Sell'
        else:
            return 'Buy'

    columns = [f'lag_{lag}' for lag in reversed(range(0, nlags))]
    columns += ['out']

    # lags
    for lag in range(0, nlags):
        df_stock[f'lag_{lag}'] = df_stock['close'].shift(lag)
    df_stock['next'] = df_stock['close'].shift(-1)
    df_stock['out'] = df_stock.apply(tagger, axis=1)
    df_clean = df_stock[columns].dropna(axis=0)
    
    # Add time features
    df_time = pd.DataFrame(index=df_clean.index)
    df_time['day_of_year'] = pd.to_datetime(df_clean.index)
    rbf = RepeatingBasisFunction(n_periods=12, column= 'day_of_year', input_range=(1,365), remainder="drop")
    rbf.fit(df_time)
    df_time = pd.DataFrame(index=df_clean.index, data=rbf.transform(df_time))  
    df_clean = pd.merge(df_clean, df_time, left_index=True, right_index=True)
    df_clean.dropna(inplace=True)

    # moving average
    ma_day = [10, 20, 50]
    for ma in ma_day:
        column_name = f"MA_{ma}"
        df_clean[column_name] = df_clean['lag_0'].rolling(ma).mean()
    
    # month and weekday
    #df_clean['month'] = df_clean.index.month
    #df_clean['weekday'] = df_clean.index.weekday
    #df_clean = pd.concat([df_clean, pd.get_dummies(df_clean['month'], drop_first=True, prefix="month")], axis=1)
    #df_clean = pd.concat([df_clean, pd.get_dummies(df_clean['weekday'], drop_first=True, prefix="weekday")], axis=1)
    #df_clean.dropna(inplace=True)


    return df_clean


def create_X_Y(df_lags):
    #df_lags = df_lags[df_lags.index < pd.to_datetime('06/01/2022')]
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
