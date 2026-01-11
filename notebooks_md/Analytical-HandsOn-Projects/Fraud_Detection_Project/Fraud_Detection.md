## Data
- Suggested: Credit Card Fraud (Kaggle)
- Required columns: features + `Class` binary target
- Place data under `data/` and set `DATA_PATH` below.


```python
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path('data') / 'creditcard.csv'  # update to your file
df = pd.read_csv(DATA_PATH)
df.head()
```

## 1. EDA & Prep
- Class balance check
- Missing values/outliers
- Train/validation split (stratified)


```python
target = 'Class'
X = df.drop(columns=[target])
y = df[target]

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
y_train.value_counts(normalize=True)
```

## 2. Baseline Model
- Use class weights or sampling
- Start with Logistic Regression / RandomForest
- Metrics: ROC AUC, PR AUC, recall at fixed FPR


```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
])
model.fit(X_train, y_train)
proba = model.predict_proba(X_valid)[:,1]
print('ROC AUC', roc_auc_score(y_valid, proba))
print('PR AUC', average_precision_score(y_valid, proba))
```

## 3. Thresholding & Monitoring
- Choose threshold for business objective
- Calibrate if needed
- Monitor drift and alerting
