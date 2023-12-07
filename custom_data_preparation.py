# to read in 2 prepared input files, randomize them, and output to a new file
import pandas


def print_if_short(text):
    if len(text) < 1000:
        print("SHORT:", text)


# both of these are roughly 1.4K each
# although it may be beneficial to have the model lean more towards 60% AI, 40% AI (for false positives?)
human_text = pandas.read_csv("data/custom/train_essays.csv")
ai_text = pandas.read_csv("data/custom/Mistral7B_CME_v2.csv")

aigt = pandas.read_csv("data/daigt/train_drcat_04.csv")
concat = pandas.concat([human_text, ai_text])

concat = concat.drop('prompt_name', axis=1)

# split the full file (I think)
train = aigt.sample(random_state=1, frac=.8)
# make sure to drop the generated column from the test data
test = aigt.iloc[::-1].drop('generated', axis=1).sample(random_state=1, frac=.2)

# or is the issue happening here?
pandas.DataFrame(train).to_csv('data/custom/combined/daigt-train.csv', index=False)
pandas.DataFrame(test).to_csv('data/custom/combined/daigt-test.csv', index=False)
