# just a practice file for reading in csv files w/ pandas and looking at it
import nltk
import pandas
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression


# note: we'll want to swap these features out for ones that are more relevant/useful
def get_character_count(text):
    return len(text)


def get_word_count(text):
    return len(text.split())


def count_unique_words(text):
    return len(set(text.split()))


def count_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    stopwords_x = [w for w in word_tokens if w in stop_words]
    return len(stopwords_x)


def count_capital_words(text):
    return sum(map(str.isupper,text.split()))


def count_sentences(text):
    return len(nltk.sent_tokenize(text))


def generate_features(dataframe):
    dataframe['char_count'] = dataframe['text'].apply(lambda x: get_character_count(x))
    dataframe['word_count'] = dataframe['text'].apply(lambda x: get_word_count(x))
    dataframe['unique_words'] = dataframe['text'].apply(lambda x: count_unique_words(x))
    dataframe['stop_words'] = dataframe['text'].apply(lambda x: count_unique_words(x))
    dataframe['capital_words'] = dataframe['text'].apply(lambda x: count_capital_words(x))
    dataframe['sentence_count'] = dataframe['text'].apply(lambda x: count_sentences(x))
    dataframe['average_word_length'] = dataframe['char_count'] / dataframe['word_count']
    dataframe['average_sentence_length'] = dataframe['word_count'] / dataframe['sentence_count']
    dataframe['stopwords_vs_words'] = dataframe['stop_words'] / dataframe['word_count']

    return dataframe


# removes all columns not related to the features or the target
def drop_non_features(dataframe):
    dataframe = dataframe.drop(['id', 'prompt_id', 'text'], axis=1)

    return dataframe

# headers required: id, prompt_id, text, generated
# note: assuming that 'train' and 'test' names are critical here for kaggle to be able to swap during evaluation
train = pandas.read_csv('data/competition/train_essays.csv')
test = pandas.read_csv('data/competition/test_essays.csv')

# generate features for training set
train_with_features = generate_features(train)
train_with_features = drop_non_features(train_with_features)
test_with_features = generate_features(test)
test_with_features = drop_non_features(test_with_features)

# split features out from targets for both train/test
features_train = train_with_features.loc[:, train_with_features.columns != 'generated']
target_train = train_with_features['generated']

# let's use simple logistic regression here for the first algorithm
clf = LogisticRegression(random_state=0, max_iter=100).fit(features_train, target_train)

print(clf.predict(test_with_features))

prediction_probabilities = clf.predict_proba(test_with_features)[:, 0]
print("Prediction Probabilities:", prediction_probabilities)

# output file expects the id of the prompt, plus the probability of it being generated
pandas.DataFrame({'id': test["id"], 'generated': prediction_probabilities}).to_csv('submission.csv', index=False)