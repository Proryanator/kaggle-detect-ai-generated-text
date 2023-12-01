import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report

train = pd.read_csv('/kaggle/input/daigt-proper-train-dataset/train_drcat_04.csv')
test = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')
train.rename(columns={'essay_id': 'id',
                      'label': 'generated',
                      'prompt': 'prompt_id'}, inplace=True)
train['prompt_id'] = pd.factorize(train['prompt_id'])[0]
train = train[['id', 'prompt_id', 'text', 'generated']]
train
df = pd.concat([train['text'], test['text']], axis=0)
vectorizer = TfidfVectorizer(stop_words='english', max_features=50000)
X = vectorizer.fit_transform(df)
lr_model = LogisticRegression()
cv = StratifiedKFold(n_splits=5, shuffle=True)
auc_scores = []
for train_idx, val_idx in cv.split(X[:train.shape[0]], train['generated']):
    X_train, X_val = X[:train.shape[0]][train_idx], X[:train.shape[0]][val_idx]
    y_train, y_val = train['generated'].iloc[train_idx], train['generated'].iloc[val_idx]

    lr_model.fit(X_train, y_train)

    preds_val_lr = lr_model.predict_proba(X_val)[:, 1]

    auc_score = roc_auc_score(y_val, preds_val_lr)
    auc_scores.append(auc_score)
for i, score in enumerate(auc_scores, 1):
    print(f'ROC AUC for fold {i}: {score:.4f}')
print('Average ROC AUC:', round(sum(auc_scores) / len(auc_scores), 4))
print('Standard deviation:',
      round((sum([(x - sum(auc_scores) / len(auc_scores)) ** 2 for x in auc_scores]) / len(auc_scores)) ** 0.5, 4))
xgb_model = XGBClassifier()
cv = StratifiedKFold(n_splits=5, shuffle=True)
auc_scores = []
for train_idx, val_idx in cv.split(X[:train.shape[0]], train['generated']):
    X_train, X_val = X[:train.shape[0]][train_idx], X[:train.shape[0]][val_idx]
    y_train, y_val = train['generated'].iloc[train_idx], train['generated'].iloc[val_idx]

    xgb_model.fit(X_train, y_train)

    preds_val_xgb = xgb_model.predict_proba(X_val)[:, 1]

    auc_score = roc_auc_score(y_val, preds_val_xgb)
    auc_scores.append(auc_score)
for i, score in enumerate(auc_scores, 1):
    print(f'ROC AUC for fold {i}: {score:.4f}')
print('Average ROC AUC:', round(sum(auc_scores) / len(auc_scores), 4))
print('Standard deviation:',
      round((sum([(x - sum(auc_scores) / len(auc_scores)) ** 2 for x in auc_scores]) / len(auc_scores)) ** 0.5, 4))
ensemble = VotingClassifier(estimators=[('lr', lr_model), ('xgb', xgb_model)], voting='soft')
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_val)
print(classification_report(y_val, y_pred))
print(f'Accuracy: {roc_auc_score(y_val, y_pred)}\n')
preds_train = ensemble.predict_proba(X[:train.shape[0]])[:, 1]
preds_test = ensemble.predict_proba(X[train.shape[0]:])[:, 1]
print('ROC AUC train:', roc_auc_score(train['generated'], preds_train))
pd.DataFrame({'id': test["id"], 'generated': preds_test}).to_csv('submission.csv', index=False)
