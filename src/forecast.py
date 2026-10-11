import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

LAGS = (1, 24, 48, 168)
FEATURES = [f'lag_{n}' for n in LAGS] + ['hour_sin','hour_cos','dow_sin','dow_cos','month_sin','month_cos','weekend']

def validate_frame(df):
    required = {'timestamp','demand_mw'}
    if not required.issubset(df.columns):
        raise ValueError('CSV must contain timestamp and demand_mw columns.')
    data = df[['timestamp','demand_mw']].copy()
    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce', utc=True)
    data['demand_mw'] = pd.to_numeric(data['demand_mw'], errors='coerce')
    if data.isna().any().any():
        raise ValueError('Missing/invalid timestamps or demand values detected. Please clean the file first.')
    if (data['demand_mw'] < 0).any():
        raise ValueError('Demand cannot be negative.')
    if data['timestamp'].duplicated().any():
        raise ValueError('Duplicate timestamps detected.')
    data = data.sort_values('timestamp').set_index('timestamp')
    if len(data) < 24*21:
        raise ValueError('At least 21 days of hourly data are required.')
    if not (data.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError('Input must have continuous hourly timestamps in UTC, with no gaps.')
    return data

def calendar_features(idx):
    idx = pd.DatetimeIndex(idx)
    return pd.DataFrame({
        'hour_sin':np.sin(2*np.pi*idx.hour/24),
        'hour_cos':np.cos(2*np.pi*idx.hour/24),
        'dow_sin':np.sin(2*np.pi*idx.dayofweek/7),
        'dow_cos':np.cos(2*np.pi*idx.dayofweek/7),
        'month_sin':np.sin(2*np.pi*(idx.month-1)/12),
        'month_cos':np.cos(2*np.pi*(idx.month-1)/12),
        'weekend':(idx.dayofweek >= 5).astype(int),
    },index=idx)

def supervised(data):
    frame = calendar_features(data.index)
    for n in LAGS:
        frame[f'lag_{n}'] = data['demand_mw'].shift(n)
    frame['target'] = data['demand_mw']
    return frame.dropna()

def fit_evaluate(data):
    frame = supervised(data)
    cut = int(len(frame)*0.8)
    if cut < 100 or len(frame)-cut < 24:
        raise ValueError('Not enough records for training and test split.')
    train, test = frame.iloc[:cut], frame.iloc[cut:]
    model = RandomForestRegressor(n_estimators=120, max_depth=14, min_samples_leaf=3, random_state=42, n_jobs=-1)
    model.fit(train[FEATURES],train['target'])
    pred = model.predict(test[FEATURES])
    baseline = test['lag_24'].to_numpy()
    actual = test['target'].to_numpy()
    def metrics(values):
        return {'MAE (MW)':round(float(mean_absolute_error(actual,values)),2),
                'RMSE (MW)':round(float(np.sqrt(mean_squared_error(actual,values))),2),
                'R²':round(float(r2_score(actual,values)),3)}
    result = pd.DataFrame({'Actual':actual,'Random Forest':pred,'Seasonal naive (24h)':baseline},index=test.index)
    return model,result,pd.DataFrame({'Random Forest':metrics(pred),'Seasonal naive':metrics(baseline)}).T

def forecast_next_24(model,data):
    history = data['demand_mw'].copy()
    dates = pd.date_range(history.index[-1]+pd.Timedelta(hours=1),periods=24,freq='h',tz='UTC')
    outputs = []
    for ts in dates:
        row = calendar_features(pd.DatetimeIndex([ts]))
        for n in LAGS:
            row[f'lag_{n}'] = float(history.loc[ts-pd.Timedelta(hours=n)])
        value = max(0.,float(model.predict(row[FEATURES])[0]))
        history.loc[ts] = value
        outputs.append(value)
    return pd.DataFrame({'Forecast demand (MW)':outputs},index=dates)

def sample_data(days=100,seed=42):
    rng=np.random.default_rng(seed)
    idx=pd.date_range('2025-01-01',periods=days*24,freq='h',tz='UTC')
    h=idx.hour.to_numpy(); dow=idx.dayofweek.to_numpy()
    demand=3200+580*np.sin(2*np.pi*(h-8)/24)+240*np.sin(4*np.pi*(h-7)/24)-260*(dow>=5)+140*np.sin(np.arange(len(idx))*2*np.pi/(24*7))+rng.normal(0,90,len(idx))
    return pd.DataFrame({'demand_mw':np.maximum(demand,500)},index=idx)

def evaluate_24h_origins(model, data, origins=6):
    """Evaluate recursive 24h forecasts at origins within the holdout period.

    The model is trained once on the initial 80% of supervised observations.
    Only observations at or before each origin are available to the forecaster.
    """
    frame = supervised(data)
    cut = int(len(frame) * .8)
    test_start = frame.index[cut]
    eligible = data.index[(data.index >= test_start - pd.Timedelta(hours=1)) &
                          (data.index <= data.index[-1] - pd.Timedelta(hours=24))]
    if len(eligible) < 1:
        raise ValueError('Not enough holdout history for 24-hour validation.')
    positions = np.unique(np.linspace(0, len(eligible)-1, min(origins, len(eligible)), dtype=int))
    rows = []
    for pos in positions:
        origin = eligible[pos]
        history = data.loc[:origin]
        future = forecast_next_24(model, history)
        actual = data.loc[future.index, 'demand_mw']
        for ts, observed, predicted in zip(future.index, actual, future.iloc[:, 0]):
            rows.append({'origin_utc': origin, 'timestamp': ts,
                         'horizon_h': int((ts-origin)/pd.Timedelta(hours=1)),
                         'actual_mw': float(observed), 'forecast_mw': float(predicted),
                         'seasonal_naive_mw': float(history.loc[ts-pd.Timedelta(hours=24), 'demand_mw'])})
    result = pd.DataFrame(rows)
    def score(col):
        return {'MAE (MW)': round(float(mean_absolute_error(result.actual_mw, result[col])), 2),
                'RMSE (MW)': round(float(np.sqrt(mean_squared_error(result.actual_mw, result[col]))), 2)}
    scores = pd.DataFrame({'Recursive Random Forest': score('forecast_mw'),
                           'Seasonal naive (24h)': score('seasonal_naive_mw')}).T
    return result, scores
