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
