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

| Dataset             | Original Size | Deduplicated Size | Time (seconds) |
|---------------------|--------------:|-------------------:|---------------:|
| bbc | 1225 | 1144 | 0.26         |
| senteval_cr | 3012 | 2990 | 0.12         |
| tweet_sentiment_extraction | 27481 | 26695 | 1.63         |
| emotion | 16000 | 15695 | 0.68         |
| amazon_counterfactual | 5000 | 4992 | 0.33         |
| ag_news | 120000 | 106921 | 4.39         |
| enron_spam | 31716 | 20540 | 1.65         |
| subj | 8000 | 7990 | 0.57         |
| sst5 | 8544 | 8526 | 0.57         |
| 20_newgroups | 11314 | 10684 | 0.68         |
| hatespeech_offensive | 22783 | 22090 | 0.89         |
| ade | 17637 | 15718 | 0.66         |
| imdb | 25000 | 24830 | 1.68         |
| massive_scenario | 11514 | 9366 | 0.46         |
| student | 117519 | 63858 | 4.11         |
| squad_v2 | 130319 | 115548 | 11.79         |
| wikitext | 1801350 | 884698 | 56.96         |
