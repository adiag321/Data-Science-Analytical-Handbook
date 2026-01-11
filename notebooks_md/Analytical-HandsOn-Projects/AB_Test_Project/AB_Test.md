## Data
- Suggested: Olist AB test logs or synthetic e-commerce experiment
- Required columns: `user_id`, `group`, `metric`, optionally covariates
- Place data under `data/` and set `DATA_PATH` below.


```python
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path('data') / 'ab_test.csv'  # update to your file
df = pd.read_csv(DATA_PATH)
df.head()
```

## 1. Sanity Checks
- Randomization balance on key covariates
- Traffic split and exposure counts
- Outlier handling rules


```python
df['group'].value_counts(normalize=True)
# add covariate balance checks if available
```

## 2. Effect Estimation
- Choose metric type (binary/continuous/count)
- Use parametric and nonparametric intervals
- Multiple comparisons if multi-variant


```python
control = df[df['group'] == 'control']['metric']
treatment = df[df['group'] == 'treatment']['metric']
ate = treatment.mean() - control.mean()

# Bootstrap CI
rng = np.random.default_rng(42)
boot = [rng.choice(treatment, size=len(treatment), replace=True).mean() -
        rng.choice(control, size=len(control), replace=True).mean()
        for _ in range(2000)]
ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
ate, ci_low, ci_high
```

## 3. Nonparametric Test
- Mann-Whitney U for continuous, Fisher/Chi2 for binary
- Consider CUPED if pre-experiment metric available


```python
import scipy.stats as st
u_stat, p_val = st.mannwhitneyu(treatment, control, alternative='two-sided')
u_stat, p_val
```

## 4. Readout
- Practical significance, guardrail metrics
- Power/post-hoc MDE
- Recommendation
