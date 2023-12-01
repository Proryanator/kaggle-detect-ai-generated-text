import regex as re
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

train = pd.read_csv("data/ensemble-learning/train_v2_drcat_02.csv")
test = pd.read_csv('data/ensemble-learning/test_essays.csv')

train_ = train[train.RDizzl3_seven == False].reset_index(drop=True)
train_ = train[train["label"]==1].sample(8000)

train = train[train.RDizzl3_seven == True].reset_index(drop=True)
train = pd.concat([train, train_])
train['text'] = train['text'].str.replace('\n', '')
test['text'] = test['text'].str.replace('\n', '')
train['label'].value_counts()
df = pd.concat([train['text'], test['text']])
vectorizer = TfidfVectorizer(sublinear_tf=True,
                             ngram_range=(3, 4),
                             tokenizer=lambda x: re.findall(r'[^\W]+', x),
                             token_pattern=None,
                             strip_accents='unicode',
                             )
vectorizer = vectorizer.fit(test['text'])

X = vectorizer.transform(df)
lr_model = LogisticRegression()
sgd_model = SGDClassifier(max_iter=5000, loss="modified_huber", random_state=42)
ensemble = VotingClassifier(estimators=[('lr', lr_model),
                                        ('sgd', sgd_model),
                                       ],
                            weights=[0.01, 0.99],
                            voting='soft'
                           )
ensemble.fit(X[:train.shape[0]], train.label)
preds_test = ensemble.predict_proba(X[train.shape[0]:])[:, 1]
pd.DataFrame({'id':test["id"], 'generated':preds_test}).to_csv('submission.csv', index=False)
