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
| bbc | 1225 | 1144 | 6.61 | 0.23 |
| senteval_cr | 3012 | 2990 | 0.73 | 0.13 |
| tweet_sentiment_extraction | 27481 | 26695 | 2.86 | 1.64 |
| emotion | 16000 | 15696 | 1.90 | 0.66 |
| amazon_counterfactual | 5000 | 4992 | 0.16 | 0.31 |
| ag_news | 120000 | 106921 | 10.90 | 4.21 |
| enron_spam | 31716 | 20539 | 35.24 | 1.57 |
| subj | 8000 | 7990 | 0.12 | 0.57 |
| sst5 | 8544 | 8526 | 0.21 | 0.55 |
| 20_newgroups | 11314 | 10685 | 5.56 | 0.69 |
| hatespeech_offensive | 22783 | 22090 | 3.04 | 0.84 |
| ade | 17637 | 15718 | 10.88 | 0.68 |
| imdb | 25000 | 24832 | 0.67 | 1.65 |
| massive_scenario | 11514 | 9366 | 18.66 | 0.43 |
| student | 117519 | 63865 | 45.66 | 4.18 |
| squad_v2 | 130319 | 115546 | 11.34 | 11.20 |
| wikitext | 1801350 | 884586 | 50.89 | 56.11 |


### Train/Test Deduplication Benchmark

| Dataset | Train Size | Test Size | Deduplicated Test Size | % Removed | Deduplication Time (s) |
| --- | --- | --- | --- | --- | --- |
| bbc | 1225 | 1000 | 875 | 12.50 | 0.39 |
| senteval_cr | 3012 | 753 | 750 | 0.40 | 0.11 |
| tweet_sentiment_extraction | 27481 | 3534 | 3411 | 3.48 | 0.86 |
| emotion | 16000 | 2000 | 1926 | 3.70 | 0.56 |
| amazon_counterfactual | 5000 | 5000 | 4990 | 0.20 | 0.48 |
| ag_news | 120000 | 7600 | 6197 | 18.46 | 3.38 |
| enron_spam | 31716 | 2000 | 1063 | 46.85 | 1.98 |
| subj | 8000 | 2000 | 1999 | 0.05 | 0.58 |
| sst5 | 8544 | 2210 | 2205 | 0.23 | 0.56 |
| 20_newgroups | 11314 | 7532 | 7310 | 2.95 | 2.26 |
| hatespeech_offensive | 22783 | 2000 | 1926 | 3.70 | 0.72 |
| ade | 17637 | 5879 | 4954 | 15.73 | 0.82 |
| imdb | 25000 | 25000 | 24805 | 0.78 | 2.65 |
| massive_scenario | 11514 | 2974 | 2188 | 26.43 | 0.43 |
| student | 117519 | 5000 | 2395 | 52.10 | 3.02 |
| squad_v2 | 130319 | 11873 | 11869 | 0.03 | 9.11 |
| wikitext | 1801350 | 4358 | 3610 | 17.16 | 36.10 |
