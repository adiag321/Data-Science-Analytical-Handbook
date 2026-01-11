## Data
- Suggested: Store Item Demand Forecasting (Kaggle)
- Required columns: `date`, `store`, `item`, `sales`
- Place data under `data/` and set `DATA_PATH` below.


```python
import pandas as pd
from pathlib import Path

DATA_PATH = Path('data') / 'train.csv'  # update to your file
df = pd.read_csv(DATA_PATH, parse_dates=['date'])
df.head()
```

## 1. Prep
- Sort by date, ensure no duplicates
- Create time-based splits (train/val/test)
- Optional: aggregate by store/item hierarchy
- Feature hints: lags, rolling means, promos, holidays


```python
df = df.sort_values('date')
# example split
split_date = df['date'].quantile(0.8)
train = df[df['date'] <= split_date]
valid = df[df['date'] > split_date]
train.shape, valid.shape
```

## 2. Baselines
- Naive (last value)
- Moving average
- Seasonal naive if seasonality known


```python
# simple rolling mean baseline per store-item
window = 7
train = train.set_index('date')
valid = valid.set_index('date')
baseline = (train.groupby(['store','item'])['sales']
                    .rolling(window).mean()
                    .reset_index()
                    .rename(columns={'sales':'forecast'}))
# join to validation and compute error
```

## 3. Models
- Prophet/ETS/ARIMA for univariate
- Gradient boosting / deep nets for global models
- Hierarchical reconciliation if needed


```python
from sklearn.metrics import mean_absolute_error
# placeholder for model training; plug in your model
# y_true = ...
# y_pred = ...
# mean_absolute_error(y_true, y_pred)
```

## 4. Evaluation
- Use time-aware CV (rolling origin)
- Metrics: MAE, RMSE, MAPE, WAPE
- Business readout: stockouts avoided, overstock cost
