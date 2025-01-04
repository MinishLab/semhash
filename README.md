# SemHash: Fast Text Deduplication using Semantic Hashing

SemHash is a technique to efficiently deduplicate datasets based on semantic similarity.

## Table of Contents
- [Quickstart](#quickstart)

## Quickstart

Install the package with:
```bash
pip install semhash
```

Deduplicate a single dataset with the following code:

```python
from model2vec import StaticModel
from semhash import SemHash

# Load an embedding model
model = StaticModel.from_pretrained("minishlab/potion-base-8M")

# Initialize a SemHash with the model
semhash = SemHash(model=model)

# Create some texts to deduplicate
texts = [
        "It's dangerous to go alone!",
        "It's dangerous to go alone!",  # Exact duplicate
        "It's not safe to go by yourself!",  # Semantically similar
]

# Deduplicate the texts
deduplicated_texts = semhash.fit_deduplicate(records=texts, threshold=0.5)
```


Or, deduplicate across two datasets (for example a train and test set) with the following code:

```python
from model2vec import StaticModel
from semhash import SemHash

# Load an embedding model
model = StaticModel.from_pretrained("minishlab/potion-base-8M")

# Initialize a SemHash with the model
semhash = SemHash(model=model)

# Create some texts to deduplicate
train = [
    "It's dangerous to go alone!",
    "It's a secret to everybody.",
    "Ganondorf has invaded Hyrule!",
]
test = [
    "It's dangerous to go alone!",  # Exact duplicate
    "It's not safe to go by yourself!",  # Semantically similar
    "The master sword seals the darkness",
]

# Fit on the training data
semhash.fit(records=train)
# Deduplicate the test data against the training data
deduplicated_texts = semhash.deduplicate(records=test, threshold=0.5)
```


## Benchmarks

### Train Deduplication Benchmark

| Dataset | Original Train Size | Deduplicated Train Size | % Removed | Deduplication Time (s) |
| --- | --- | --- | --- | --- |
| bbc | 1225 | 1144 | 6.61 | 0.28 |
| senteval_cr | 3012 | 2990 | 0.73 | 0.14 |
| tweet_sentiment_extraction | 27481 | 26695 | 2.86 | 1.68 |
| emotion | 16000 | 15695 | 1.91 | 0.70 |
| amazon_counterfactual | 5000 | 4992 | 0.16 | 0.33 |
| ag_news | 120000 | 106921 | 10.90 | 4.67 |
| enron_spam | 31716 | 20539 | 35.24 | 1.68 |
| subj | 8000 | 7990 | 0.12 | 0.60 |
| sst5 | 8544 | 8526 | 0.21 | 0.62 |
| 20_newgroups | 11314 | 10672 | 5.67 | 0.74 |
| hatespeech_offensive | 22783 | 22090 | 3.04 | 0.92 |
| ade | 17637 | 15718 | 10.88 | 0.71 |
| imdb | 25000 | 24830 | 0.68 | 1.79 |
| massive_scenario | 11514 | 9366 | 18.66 | 0.44 |
| student | 117519 | 63868 | 45.65 | 4.38 |
| squad_v2 | 130319 | 115550 | 11.33 | 12.32 |
| wikitext | 1801350 | 884640 | 50.89 | 59.78 |


### Train/Test Deduplication Benchmark

| Dataset | Train Size | Test Size | Deduplicated Test Size | % Removed | Deduplication Time (s) |
| --- | --- | --- | --- | --- |
| bbc | 1225 | 1000 | 873 | 1.3e+01 | 0.42 |
| senteval_cr | 3012 | 753 | 750 | 0.4 | 0.12 |
| tweet_sentiment_extraction | 27481 | 3534 | 3411 | 3.5 | 0.91 |
| emotion | 16000 | 2000 | 1927 | 3.6 | 0.59 |
| amazon_counterfactual | 5000 | 5000 | 4990 | 0.2 | 0.53 |
| ag_news | 120000 | 7600 | 6197 | 1.8e+01 | 3.59 |
| enron_spam | 31716 | 2000 | 1066 | 4.7e+01 | 2.09 |
| subj | 8000 | 2000 | 1999 | 0.05 | 0.62 |
| sst5 | 8544 | 2210 | 2205 | 0.23 | 0.60 |
| 20_newgroups | 11314 | 7532 | 7311 | 2.9 | 2.32 |
| hatespeech_offensive | 22783 | 2000 | 1925 | 3.7 | 0.71 |
| ade | 17637 | 5879 | 4957 | 1.6e+01 | 0.90 |
| imdb | 25000 | 25000 | 24804 | 0.78 | 2.99 |
| massive_scenario | 11514 | 2974 | 2187 | 2.6e+01 | 0.44 |
| student | 117519 | 5000 | 2398 | 5.2e+01 | 3.22 |
| squad_v2 | 130319 | 11873 | 11869 | 0.034 | 9.88 |
| wikitext | 1801350 | 4358 | 2153 | 5.1e+01 | 37.79 |
