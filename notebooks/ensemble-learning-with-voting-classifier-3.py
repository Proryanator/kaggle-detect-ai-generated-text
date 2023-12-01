# ## ## *# *# I# n# t# r# o# d# u# c# t# i# o# n# *# *# <# a# c# l# a# s# s# =# "# a# n# c# h# o# r# "# i# d# =# "# h# 1# "# ># <# /# a# ># <# d# i# v# c# l# a# s# s# =# "# a# l# e# r# t# a# l# e# r# t# -# i# n# f# o# "# ># W# e# l# c# o# m# e# t# o# t# h# i# s# K# a# g# g# l# e# n# o# t# e# b# o# o# k# .# I# n# t# h# i# s# i# t# e# r# a# t# i# o# n# ,# w# e# c# o# n# t# i# n# u# e# o# u# r# e# x# p# l# o# r# a# t# i# o# n# i# n# t# o# t# h# e# f# a# s# c# i# n# a# t# i# n# g# d# o# m# a# i# n# o# f# <# b# ># e# s# s# a# y# c# l# a# s# s# i# f# i# c# a# t# i# o# n# <# /# b# ># .# T# h# e# p# r# i# m# a# r# y# o# b# j# e# c# t# i# v# e# i# s# t# o# r# e# f# i# n# e# t# h# e# c# l# a# s# s# i# f# i# c# a# t# i# o# n# t# a# s# k# ,# d# i# s# t# i# n# g# u# i# s# h# i# n# g# b# e# t# w# e# e# n# e# s# s# a# y# s# <# b# ># a# u# t# h# o# r# e# d# b# y# s# t# u# d# e# n# t# s# <# /# b# ># a# n# d# t# h# o# s# e# <# b# ># g# e# n# e# r# a# t# e# d# b# y# a# l# a# n# g# u# a# g# e# m# o# d# e# l# (# L# L# M# )# <# /# b# ># .# L# e# v# e# r# a# g# i# n# g# e# n# s# e# m# b# l# e# l# e# a# r# n# i# n# g# ,# w# e# i# n# t# r# o# d# u# c# e# a# <# b# ># S# o# f# t# V# o# t# i# n# g# C# l# a# s# s# i# f# i# e# r# <# /# b# ># t# h# a# t# c# o# m# b# i# n# e# s# t# h# e# c# a# p# a# b# i# l# i# t# i# e# s# o# f# <# b# ># L# o# g# i# s# t# i# c# R# e# g# r# e# s# s# i# o# n# <# /# b# ># a# n# d# <# b# ># S# t# o# c# h# a# s# t# i# c# G# r# a# d# i# e# n# t# D# e# s# c# e# n# t# (# S# G# D# )# <# /# b# ># m# o# d# e# l# s# .# <# /# d# i# v# >

# ## ## ## 📋# *# *# T# a# b# l# e# o# f# C# o# n# t# e# n# t# s# *# *# *# [# I# n# t# r# o# d# u# c# t# i# o# n# ]# (# ## h# 1# )# *# [# I# m# p# o# r# t# L# i# b# r# a# r# i# e# s# ]# (# ## h# 2# )# *# [# R# e# a# d# D# a# t# a# s# e# t# ]# (# ## h# 3# )# *# [# V# e# c# t# o# r# i# z# e# T# e# x# t# D# a# t# a# ]# (# ## h# 4# )# *# [# D# e# f# i# n# e# a# n# d# T# r# a# i# n# M# o# d# e# l# s# ]# (# ## h# 5# )# *# [# G# e# n# e# r# a# t# e# P# r# e# d# i# c# t# i# o# n# s# a# n# d# C# r# e# a# t# e# S# u# b# m# i# s# s# i# o# n# F# i# l# e# ]# (# ## h# 6# )

# ## ## 📚# *# *# I# m# p# o# r# t# L# i# b# r# a# r# i# e# s# *# *# <# a# c# l# a# s# s# =# "# a# n# c# h# o# r# "# i# d# =# "# h# 2# "# ># <# /# a# >

import regex as re
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# ## ## 📖# *# *# R# e# a# d# D# a# t# a# s# e# t# *# *# <# a# c# l# a# s# s# =# "# a# n# c# h# o# r# "# i# d# =# "# h# 3# "# ># <# /# a# >

train = pd.read_csv("/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv")
test = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')

train_ = train[train.RDizzl3_seven == False].reset_index(drop=True)
train_ = train[train["label"]==1].sample(8000)
train = train[train.RDizzl3_seven == True].reset_index(drop=True)
train = pd.concat([train, train_])
train['text'] = train['text'].str.replace('\n', '')
test['text'] = test['text'].str.replace('\n', '')
train['label'].value_counts()

# ## ## 🔠# *# *# V# e# c# t# o# r# i# z# e# T# e# x# t# D# a# t# a# *# *# <# a# c# l# a# s# s# =# "# a# n# c# h# o# r# "# i# d# =# "# h# 4# "# ># <# /# a# >

%%time
df = pd.concat([train['text'], test['text']])

vectorizer = TfidfVectorizer(sublinear_tf=True,
                             ngram_range=(3, 4),
                             tokenizer=lambda x: re.findall(r'[^\W]+', x),
                             token_pattern=None,
                             strip_accents='unicode',
                             )

vectorizer = vectorizer.fit(test['text'])
X = vectorizer.transform(df)

# ## ## ⚙# ️# *# *# D# e# f# i# n# e# a# n# d# T# r# a# i# n# M# o# d# e# l# s# *# *# <# a# c# l# a# s# s# =# "# a# n# c# h# o# r# "# i# d# =# "# h# 5# "# ># <# /# a# >

%%time
lr_model = LogisticRegression()
sgd_model = SGDClassifier(max_iter=5000, loss="modified_huber", random_state=42)

ensemble = VotingClassifier(estimators=[('lr', lr_model),
                                        ('sgd', sgd_model),
                                       ],
                            weights=[0.01, 0.99],
                            voting='soft'
                           )
ensemble.fit(X[:train.shape[0]], train.label)

# ## ## 📊# *# *# G# e# n# e# r# a# t# e# P# r# e# d# i# c# t# i# o# n# s# a# n# d# C# r# e# a# t# e# S# u# b# m# i# s# s# i# o# n# F# i# l# e# *# *# <# a# c# l# a# s# s# =# "# a# n# c# h# o# r# "# i# d# =# "# h# 6# "# ># <# /# a# >

preds_test = ensemble.predict_proba(X[train.shape[0]:])[:, 1]
pd.DataFrame({'id':test["id"], 'generated':preds_test}).to_csv('submission.csv', index=False)