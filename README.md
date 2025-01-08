
<div align="center">

# SemHash: Fast Semantic Text Deduplication


  <h2>
    <a href="https://pypi.org/project/semhash/"><img src="https://img.shields.io/pypi/v/semhash?color=%23007ec6&label=pypi%20package" alt="Package version"></a>
    <a href="https://pypi.org/project/semhash/"><img src="https://img.shields.io/pypi/pyversions/semhash" alt="Supported Python versions"></a>
    <a href="https://app.codecov.io/gh/MinishLab/semhash">
    <img src="https://codecov.io/gh/MinishLab/semhash/graph/badge.svg?token=YPOD6HD0MG" alt="Codecov">
    </a>
    <a href="https://github.com/MinishLab/semhash/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License - MIT"></a>
  </h2>


[Quickstart](#quickstart) •
[Main Features](#main-features) •
[Usage](#usage) •
[Benchmarks](#benchmarks)

</div>


SemHash is a technique to efficiently deduplicate datasets based on semantic similarity. It uses a combination of lightning-fast embeddings through [model2vec](https://github.com/MinishLab/model2vec) and ANN-based similarity search through [vicinity](https://github.com/MinishLab/vicinity).



## Quickstart

Install the package with:
```bash
pip install semhash
```

Deduplicate a single dataset with the following code (note: this example assumes you have `datasets` installed, which you can install with `pip install datasets`):

```python
from datasets import load_dataset
from semhash import SemHash

# Load a dataset to deduplicate
texts = load_dataset("ag_news", split="train")["text"]

# Initialize a SemHash instance
semhash = SemHash.from_records(records=texts)

# Deduplicate the texts
result = semhash.self_deduplicate(records=texts)

# Check the texts
result.deduplicated
# Check the duplicates
result.duplicates
# See how many texts were duplicates
result.duplicate_ratio
# See how many were exact duplicates
result.exact_duplicate_ratio

```

Or, deduplicate across two datasets with the following code (eliminating train/test leakage):

```python
from datasets import load_dataset
from semhash import SemHash

# Load two datasets to deduplicate
train_texts = load_dataset("ag_news", split="train")["text"]
test_texts = load_dataset("ag_news", split="test")["text"]

# Initialize a SemHash instance with the training data
semhash = SemHash.from_records(records=train_texts)

# Deduplicate the test data against the training data
deduplicated_test_texts = semhash.deduplicate(records=test_texts).deduplicated
```

As seen above, the `deduplicate` and `self_deduplicate` functions return a `DeduplicationResult`. This object stores the deduplicated corpus, and a set of duplicate objects, along with the objects that caused duplication. For example:

```python
from datasets import load_dataset
from semhash import SemHash

# Load a dataset to deduplicate
texts = load_dataset("ag_news", split="train")["text"]

# Initialize a SemHash instance
semhash = SemHash.from_records(records=texts)

# Deduplicate the texts
result = semhash.self_deduplicate(records=texts, threshold=0.99)

for duplicate in result.duplicates:
  print("RECORD:")
  print(duplicate.record)
  if duplicate.exact:
    print("Exact match!")
  else:
    print("DUPLICATES:")
    for corpus_duplicate in duplicate.duplicates:
      print(corpus_duplicate)
  print("-" * 25)

```

For more advanced usage, you can also deduplicate across multiple datasets, or deduplicate multi-column datasets. Examples are provided in the [usage](#usage) section.

NOTE: By default, we use the ANN (approximate-nearest neighbors) backend for deduplication. We recommend keeping this since the recall for smaller datasets is ~100%, and it's needed for larger datasets (>1M samples) since these will take too long to deduplicate without ANN. If you want to use the flat/exact-matching backend, you can set `use_ann=False` in the SemHash constructor:

```python
semhash = SemHash.from_records(records=texts, use_ann=False)
```

## Main Features

- **Fast**: SemHash uses model2vec to embed texts and vicinity to perform similarity search, making it extremely fast.
- **Scalable**: SemHash can deduplicate large datasets with millions of records thanks to the ANN backends in Vicinity.
- **Flexible**: SemHash can be used to deduplicate a single dataset or across two datasets, and can also be used to deduplicate multi-column datasets (such as QA datasets).
- **Lightweight**: SemHash is a lightweight package with minimal dependencies, making it easy to install and use.

## Usage

The following examples show the various ways you can use SemHash to deduplicate datasets. These examples assume you have the `datasets` library installed, which you can install with `pip install datasets`.

<details>
<summary>  Deduplicate a single dataset </summary>
<br>

The following code snippet shows how to deduplicate a single dataset using SemHash (in this example, the train split of the [AG News dataset](https://huggingface.co/datasets/fancyzhx/ag_news)):

```python
from datasets import load_dataset
from semhash import SemHash

# Load a dataset to deduplicate
texts = load_dataset("ag_news", split="train")["text"]

# Initialize a SemHash instance
semhash = SemHash.from_records(records=texts)

# Deduplicate the texts
deduplicated_texts = semhash.self_deduplicate(records=texts)
```
</details>

<details>
<summary>  Deduplicate across two datasets </summary>
<br>

The following code snippet shows how to deduplicate across two datasets using SemHash (in this example, the train/test split of the [AG News dataset](https://huggingface.co/datasets/fancyzhx/ag_news)):

```python
from datasets import load_dataset
from semhash import SemHash

# Initialize a SemHash instance
semhash = SemHash()

# Load two datasets to deduplicate
train_texts = load_dataset("ag_news", split="train")["text"]
test_texts = load_dataset("ag_news", split="test")["text"]

# Initialize a SemHash instance
semhash = SemHash.from_records(records=train_texts)

# Deduplicate the test data against the training data
deduplicated_test_texts = semhash.deduplicate(records=test_texts)
```

</details>

<details>
<summary>  Deduplicate multi-column datasets </summary>
<br>

The following code snippet shows how to deduplicate multi-column datasets using SemHash (in this example, the train split of the QA dataset [SQuAD 2.0](https://huggingface.co/datasets/rajpurkar/squad_v2), which consists of questions, contexts, and answers):

```python
from datasets import load_dataset
from semhash import SemHash

# Load the dataset
dataset = load_dataset("squad_v2", split="train")

# Convert the dataset to a list of dictionaries
records = [
    {"context": row["context"], "question": row["question"], "answers": str(row["answers"])}
    for row in dataset
]

# Initialize SemHash with the columns to deduplicate
semhash = SemHash.from_records(records=records, columns=["question", "context", "answers"])

# Deduplicate the records
deduplicated_records = semhash.self_deduplicate(records=records)
```

</details>

<details>
<summary>  Using custom encoders </summary>
<br>

The following code snippet shows how to use a custom encoder with SemHash:

```python
from datasets import load_dataset
from model2vec import StaticModel
from semhash import SemHash

# Load a dataset to deduplicate
texts = load_dataset("ag_news", split="train")["text"]

# Load an embedding model (in this example, a multilingual model)
model = StaticModel.from_pretrained("minishlab/M2V_multilingual_output")

# Initialize a SemHash with the model and custom encoder
semhash = SemHash.from_records(records=texts, model=model)

# Deduplicate the texts
deduplicated_texts = semhash.self_deduplicate(records=texts)
```

Any encoder can be used that adheres to our [encoder protocol](https://github.com/MinishLab/semhash/blob/main/semhash/utils.py). For example, any [sentence-transformers](https://github.com/UKPLab/sentence-transformers) model can be used as an encoder:

```python
from datasets import load_dataset
from semhash import SemHash
from sentence_transformers import SentenceTransformer

# Load a dataset to deduplicate
texts = load_dataset("ag_news", split="train")["text"]

# Load a sentence-transformers model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initialize a SemHash with the model and custom encoder
semhash = SemHash.from_records(records=texts, model=model)

# Deduplicate the texts
deduplicated_texts = semhash.self_deduplicate(records=texts)
```

</details>

## Benchmarks

We've benchmarked SemHash on a variety of datasets to measure the deduplication performance and speed. The benchmarks were run with the following setup:
- The benchmarks were all run on CPU
- The benchmarks were all run with `use_ann=True`
- The used encoder is the default encoder ([potion-base-8M](https://huggingface.co/minishlab/potion-base-8M)).
- The timings include the encoding time, index building time, and deduplication time.

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


As can be seen, SemHash is extremely fast, and scales to large datasets with millions of records. There are some notable examples of train/test leakage, such as `enron_spam` and `student`, where the test dataset contains a significant amount of semantic overlap with the training dataset.

### Reproducing the Benchmarks

To run the benchmarks yourself, you can use the following command:

```bash
python -m benchmarks.run_benchmarks
```
Optionally, the datasets can be updated in the [datasets.py](https://github.com/MinishLab/semhash/blob/main/benchmarks/datasets.py) file.
