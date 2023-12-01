# just a practice file for reading in csv files w/ pandas and looking at it
import nltk
import pandas
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


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

    # shuffle the column generated to the end
    dataframe = dataframe[[col for col in dataframe.columns if col != 'generated'] + ['generated']]
    return dataframe


# removes all columns not related to the features or the target
def drop_non_features(dataframe):
    dataframe = dataframe.drop(['id', 'prompt_id', 'text'], axis=1)

    # move the column 'generated' to the end
    return dataframe


# required to download a required resource for nltk
nltk.download('punkt')

# headers required: id, prompt_id, text, generated
df = pandas.read_csv('data/competition/train_essays.csv')

# generate features and remove non-feature specific columns (leaves target)
df = generate_features(df)
df = drop_non_features(df)

# split features out from targets
features = df.loc[:, df.columns != 'generated']
target = df['generated']

# train/test data split; always in x_train, x_test, y_train, y_test
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=0)

# let's use simple logistic regression here for the first algorithm
# note: the test dataset might be too small
clf = LogisticRegression(random_state=0, max_iter=100).fit(features_train, target_train)

# run the algorithm on subset of initial test data
print(clf.predict(features_test))

print(clf.score(features_test, target_test))

# TODO: re-add prompt_id values so we can store that in the output file

# we'll need the prediction probabilities to be stored in the output file
# preds_test = clf.predict_proba(features[features_test.shape[0]:])[:, 1]

# TODO: output the file we'd need for submission
# pandas.DataFrame({'id': test["id"], 'generated': preds_test}).to_csv('submission.csv', index=False)