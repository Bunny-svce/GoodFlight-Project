# Run this to create sample CSVs to test app quickly
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

X, y = make_classification(n_samples=500, n_features=5,
                           n_classes=3, n_informative=3, random_state=42)
# train a simple classifier to produce y_pred and probs
clf = LogisticRegression(multi_class='multinomial', max_iter=1000)
clf.fit(X, y)
probs = clf.predict_proba(X)
pred = clf.predict(X)

# produce a DataFrame
df = pd.DataFrame(X, columns=[f'feat{i}' for i in range(X.shape[1])])
df['y_true'] = y
df['y_pred'] = pred
# add probability columns named prob_0 prob_1 prob_2
for i in range(probs.shape[1]):
    df[f'prob_{i}'] = probs[:, i]

df.to_csv('sample_prediction.csv', index=False)
print('Wrote sample_prediction.csv (y_true, y_pred, prob_0..prob_n).')
