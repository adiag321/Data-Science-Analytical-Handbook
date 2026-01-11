## Data
- Suggested: IBM Telco Customer Churn (Kaggle)
- Place raw data under `data/` and set `DATA_PATH` below.


```python
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path('data') / 'telco_churn.csv'  # update to your file
df = pd.read_csv(DATA_PATH)
df.head()
```

## 1. EDA
- Inspect schema, missingness, class balance
- Basic charts: churn rate, tenure distribution, services usage
- Encode/clean columns (e.g., Yes/No, total charges numeric)


```python
df.info()
df['Churn'].value_counts(normalize=True)
```

## 2. Feature Engineering
- Categorical encoding (one-hot / target)
- Numeric scaling (optional)
- Train/validation split
- Handle class imbalance (class weights or resampling)


```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression

target = 'Churn'
X = df.drop(columns=[target])
y = (df[target] == 'Yes').astype(int)

cat_cols = X.select_dtypes(include=['object']).columns
num_cols = X.select_dtypes(exclude=['object']).columns

preprocess = ColumnTransformer([
```


```python
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', 'passthrough', num_cols)
]},{
:
,
:null,
:{},
:[],
:[
```


```python
])
```


```python
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model.fit(X_train, y_train)
pred_proba = model.predict_proba(X_valid)[:, 1]
print('ROC AUC', roc_auc_score(y_valid, pred_proba))
print('AP', average_precision_score(y_valid, pred_proba))
```

## 3. Threshold & Calibration
- Plot PR curve, choose operating threshold
- Calibrate probabilities if needed (Platt/Isotonic)
- Segment metrics by key attributes (e.g., tenure, contract type)

## 4. Business Readout
- Top drivers (coefficients / SHAP)
- Expected saves from retention offer
- Risks and ethical considerations
