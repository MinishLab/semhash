
<div align="center">

# SemHash: Fast Semantic Text Deduplication


[Quickstart](#quickstart) •
[Main Features](#main-features) •
[Usage](#usage)

</div>


SemHash is a technique to efficiently deduplicate datasets based on semantic similarity. It uses a combination of lightning-fast embeddings through [model2vec](https://github.com/MinishLab/model2vec) and ANN-based similarity search through [vicinity](https://github.com/MinishLab/vicinity).



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

For more advanced usage, you can also deduplicate across multiple datasets, or deduplicate multi-column datasets. Examples are provided in the [usage](#usage) section.


## Main Features

- **Fast**: SemHash uses model2vec to embed texts and vicinity to perform similarity search, making it extremely fast.
- **Scalable**: SemHash can deduplicate large datasets with millions of records thanks to the ANN backends in Vicinity.
- **Flexible**: SemHash can be used to deduplicate a single dataset or across two datasets, and can also be used to deduplicate multi-column datasets (such as QA datasets).

## Usage

<details>
<summary>  Deduplicate a single dataset </summary>
<br>

The following code snippet shows how to deduplicate a single dataset using SemHash:

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
</details>

<details>
<summary>  Deduplicate across two datasets </summary>
<br>

The following code snippet shows how to deduplicate across two datasets using SemHash (in this example, a training and test dataset):

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

</details>

<details>
<summary>  Deduplicate multi-column datasets </summary>
<br>

The following code snippet shows how to deduplicate multi-column datasets using SemHash (in this example, a QA dataset with questions, contexts, and answers):

```python
from model2vec import StaticModel
from semhash import SemHash

# Load an embedding model
model = StaticModel.from_pretrained("minishlab/potion-base-8M")

# Initialize a SemHash with the model and columns to deduplicate
semhash = SemHash(model=model, columns=["question", "context", "answer"])

# Create some texts to deduplicate
records = [
    {"question": "What is the hero's name?", "context": "The hero is Link", "answer": "Link"},
    {"question": "What is the hero's name?", "context": "The hero is Link", "answer": "Link"},  # Exact duplicate
    {
        "question": "Who is the protagonist?",
        "context": "In this story, Link is the hero",
        "answer": "Link",
    },  # Semantically similar
    {"question": "Who is the princess?", "context": "The princess is Zelda", "answer": "Zelda"},
]

# Deduplicate the records
deduplicated_records = semhash.fit_deduplicate(records=records, threshold=0.5)
```

</details>
