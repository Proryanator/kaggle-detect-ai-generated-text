## Discussion and Findings

### Datasets Tried

Initial dataset that we'll use to represent LLM generated text is: <a href='https://www.kaggle.com/datasets/carlmcbrideellis/llm-mistral-7b-instruct-texts?select=Mistral7B_CME_v1.csv'>here</a>, generated by Mistral 7B.

Checking dataset distribution with:

```python
print(df["generated"].value_counts(normalize=True))
```

## Converting Code to Notebook

For debugging purposes and quicker development, using .py files initially.

But before submission, we'll need to convert these files to Jupyter Notebooks. We can do this via the p2j package, install it via:

```shell
pip install p2j
```

```shell
# converts notebook to python
p2j -r notebook.ipynb

# converts python to notebook
p2j file.py 
```

Additional filtering (to remove the comment code):

```shell
grep -o '^[^#%]*' file.py
```

### Pretty Printing Dataframes in Terminal

Can use a library called tabulate to do this:

```python
print(tabulate(df.head(5), headers='keys', tablefmt='psql'))
```

### First Time Using Kaggle Competition Notes

#### Submission Errors
Had some issues with my submission being scored, but it was due to:

Not using a test dataset that is identical to the provided one (which made my code work on my own test dataset, but fail to properly generate an output file when run against the real test set).

#### Internet Access Disabled (and what that means)
Disabling internet access on the notebook doesn't necessarily mean that things that require a download won't work. So anything that needs to be downloaded in the notebook is fine (i.e. used nltk to download a tokenizer, this still worked).