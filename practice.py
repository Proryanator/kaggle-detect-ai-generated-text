# some of the features below were chosen in reference to the following article:
# https://www.sciencedirect.com/science/article/pii/S266638642300200X?via%3Dihub
import re
from statistics import stdev, mean

import nltk
import pandas
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression


def count_sentences(text):
    return len(nltk.sent_tokenize(text))


def count_words(text):
    return text.apply(lambda x: len(nltk.word_tokenize(x)))


def calculate_std_deviation_sentence_length(text):
    sentences = nltk.sent_tokenize(text)
    try:
        return stdev(map(lambda s: len(s), sentences))
    # some text that's generated may only have 1 sentence
    except Exception:
        # TODO: is returning 0 the right thing to do here?
        return 0


def calculate_sentence_length_difference_consecutively(text):
    sentences = nltk.sent_tokenize(text)
    sentence_lengths = list(map(lambda s: len(nltk.word_tokenize(s)), sentences))
    sentence_differences = []
    for index in range(len(sentence_lengths) - 1):
        sentence_differences.append(abs(sentence_lengths[index] - sentence_lengths[index + 1]))

    # for those sentences that only have 1 sentence
    try:
        return mean(sentence_differences)
    except:
        Exception
    return 0


def sentences_less_than_seven_words(text):
    sentences = nltk.sent_tokenize(text)
    word_counts = list(map(lambda x: len(nltk.word_tokenize(x)), sentences))
    return len(list(filter(lambda count: count < 7, word_counts)))


def sentences_greater_than_thirty_four_words(text):
    sentences = nltk.sent_tokenize(text)
    word_counts = list(map(lambda x: len(nltk.word_tokenize(x)), sentences))
    return len(list(filter(lambda count: count >= 34, word_counts)))


def count_literal_numeric_values(text):
    return len(re.findall(r'\b\d+\b', text))


def count_word_occurrence(text, word):
    return nltk.Counter(nltk.word_tokenize(text.lower()))[word]


def generate_word_counts_for(text, words_of_interest):
    counts = text.apply(lambda x: nltk.Counter(nltk.word_tokenize(x.lower()))).apply(
        lambda counter: get_array_of_counts(counter, words_of_interest))
    return pandas.DataFrame(list(counts), columns=words_of_interest)


def get_array_of_counts(counter, words_of_interest):
    word_to_count = {}
    for word in words_of_interest:
        word_to_count[word] = counter[word]

    return word_to_count


def log_zero_columns(df):
    for columnName in df:
        if (df[columnName] == 0).all():
            print("Column [", columnName,
                  "] contains all zeros, suggesting to check code or omit from the final generated feature list...")


def plot_correlation(dataframe):
    # plot correlation between features (so we can see if we need to remove any)
    # the features do not appear to be correlated
    corr = dataframe.corr()
    fig, ax = plt.subplots(figsize=(len(corr.columns), len(corr.columns)))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()


def generate_features(df):
    text = df['text']

    # AI text versus human text may contain different amounts of complex/special characters
    # note: top 2 returned all 0's on training test dataset
    df['parenthesis_count'] = text.apply(lambda x: (x.count("(")))
    df['dash_count'] = text.apply(lambda x: (x.count('-')))
    df['colon_semicolon_count'] = text.apply(lambda x: (x.count(';') + x.count(':')))
    df['question_mark_count'] = text.apply(lambda x: (x.count('?')))
    df['single_quote_count'] = text.apply(lambda x: (x.count('\'')))

    # sentence counts, lengths, and other stats may be different
    df['sentence_count'] = text.apply(lambda x: count_sentences(x))
    df['word_count'] = count_words(text)
    df['std_dev_sentence_length'] = text.apply(lambda x: calculate_std_deviation_sentence_length(x))
    # this feature seemed to be highly correlated with the std_dev_sentence_length, so we'll cut that one out
    df['sentence_length_difference_consecutive'] = text.apply(
        lambda x: calculate_sentence_length_difference_consecutively(x))
    df['sentences_with_<_11_words'] = text.apply(lambda x: sentences_less_than_seven_words(x))
    df['sentences_with_>=_34_words'] = text.apply(lambda x: sentences_greater_than_thirty_four_words(x))

    # specific words may tend to appear more in AI versus human written
    # note: we may want to choose different words based on the domain of the essay (to make this model generalized?)
    # very important that we lowercase the text before this step, as nltk's word counter does not do this for you
    words_of_interest = ['although', 'however', 'but', 'because', 'et', 'researchers', 'others']
    df.join(generate_word_counts_for(text, words_of_interest))

    # specific datapoints, numbers, number words, etc.
    df['non_word_numbers_count'] = text.apply(lambda x: count_literal_numeric_values(x))
    # df['number_words_count'] = text.apply(lambda x: count_numbers_as_words(x))

    # TODO: impute 0 values for standard deviation and difference between sentences
    # consider adding in a word to number library to count how many times word numbers are used
    # might be a useful distinguishment between AI/human
    # https://pypi.org/project/word2number/
    # TODO: we can potentially augment/use frequency distribution as well on the above words

    # TODO: if we add features for root words, or whatnot, let's add stemming, lemmatization, etc

    log_zero_columns(df)

    return df


# removes all columns not related to the features or the target
def drop_non_features(dataframe):
    dataframe = dataframe.drop(['id', 'prompt_id', 'text'], axis=1)

    return dataframe


# headers required: id, prompt_id, text, generated
# note: assuming that 'train' and 'test' names are critical here for kaggle to be able to swap during evaluation
train = pandas.read_csv('data/custom/combined/provided_train_and_mistral7bv2_training.csv')
test = pandas.read_csv('data/custom/combined/provided_train_and_mistral7bv2_testing.csv')

print('Calculating features...')
# generate features for training set
# note: recent addition of features slows this down, performance improvements needed
train_with_features = generate_features(train)
train_with_features = drop_non_features(train_with_features)

# plot_correlation(train_with_features)
test_with_features = generate_features(test)
test_with_features = drop_non_features(test_with_features)

# split features out from targets for both train/test
features_train = train_with_features.loc[:, train_with_features.columns != 'generated']
target_train = train_with_features['generated']

print('Performing learning...')
# let's use simple logistic regression here for the first algorithm
clf = LogisticRegression(random_state=0, max_iter=10000).fit(features_train, target_train)

print("Predictions: ", clf.predict(test_with_features))

prediction_probabilities = clf.predict_proba(test_with_features)[:, 1]
print("Prediction Probabilities Mean:", mean(prediction_probabilities))

# output file expects the id of the prompt, plus the probability of it being generated
pandas.DataFrame({'id': test["id"], 'generated': prediction_probabilities}).to_csv('submission.csv', index=False)
