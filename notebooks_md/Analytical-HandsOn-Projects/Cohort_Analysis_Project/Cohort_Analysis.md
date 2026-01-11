## Data
- Suggested: Open retention datasets or your product event logs
- Required columns: `user_id`, `event_date`, optionally `amount`
- Place data under `data/` and set `DATA_PATH` below.


```python
import pandas as pd
from pathlib import Path

DATA_PATH = Path('data') / 'events.csv'  # update to your file
df = pd.read_csv(DATA_PATH, parse_dates=['event_date'])
df.head()
```

## 1. Prep Cohorts
- Define cohort by first event month
- Build retention matrix (periods since cohort start)
- Optional: revenue per cohort


```python
df['event_month'] = df['event_date'].dt.to_period('M')
first_event = df.groupby('user_id')['event_month'].min().rename('cohort')
df = df.join(first_event, on='user_id')
df['period'] = (df['event_month'] - df['cohort']).apply(lambda p: p.n)
retention = (df.drop_duplicates(['user_id','period'])
              .pivot_table(index='cohort', columns='period', values='user_id', aggfunc='count'))
retention_rate = retention.div(retention[0], axis=0)
retention_rate.head()
```

## 2. Visualization
- Heatmap of retention_rate
- Cohort size bar chart
- Revenue per cohort over time (if `amount` available)


```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sns.heatmap(retention_rate, annot=True, fmt='.0%', cmap='Blues')
plt.title('User Retention by Cohort')
plt.show()
```

## 3. Insights
- Identify strong/weak cohorts and periods of drop-off
- Attribute changes (pricing, onboarding, channels)
- Next steps: re-engagement tests, onboarding improvements
