## Data
- Suggested: UCI Online Retail or synthetic price/quantity logs
- Required columns: `date`, `product_id`, `price`, `quantity`, optionally promotions
- Place data under `data/` and set `DATA_PATH` below.


```python
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path('data') / 'pricing.csv'  # update to your file
df = pd.read_csv(DATA_PATH, parse_dates=['date'])
df.head()
```

## 1. Prep
- Filter anomalies/returns
- Create time buckets (week/month)
- Normalize price/quantity, add promotion flags
- Optional: cluster products into families


```python
df['week'] = df['date'].dt.to_period('W').apply(lambda p: p.start_time)
weekly = df.groupby(['product_id','week']).agg({'price':'mean','quantity':'sum'}).reset_index()
weekly.head()
```

## 2. Elasticity Estimation
- Log-log regression: ln(Q) ~ ln(P) + controls
- Instrumental variables if endogeneity suspected
- Segment-level elasticities


```python
import statsmodels.api as sm
weekly['ln_q'] = np.log1p(weekly['quantity'])
weekly['ln_p'] = np.log(weekly['price'])
X = sm.add_constant(weekly[['ln_p']])
y = weekly['ln_q']
ols_res = sm.OLS(y, X).fit()
ols_res.summary()
```

## 3. Scenario Simulation
- Simulate revenue vs. price changes using elasticity
- Identify optimal price under constraints
- Sensitivity analysis


```python
elasticity = ols_res.params['ln_p']
base_price = weekly['price'].mean()
base_qty = weekly['quantity'].mean()
def simulate(price_change_pct):
    new_price = base_price * (1 + price_change_pct)
    expected_qty = base_qty * (new_price / base_price) ** elasticity
    return new_price * expected_qty

for pct in [-0.1, -0.05, 0, 0.05, 0.1]:
    rev = simulate(pct)
    print(f'Price change {pct:+.0%} -> revenue {rev:,.2f}')
```

## 4. Readout
- Elasticity by segment/product
- Recommended price bands
- Risks (cannibalization, competitor reaction)
