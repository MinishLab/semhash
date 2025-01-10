
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


SemHash is a lightweight and flexible tool for deduplicating datasets using semantic similarity. It combines fast embedding generation from [Model2Vec](https://github.com/MinishLab/model2vec) with efficient ANN-based similarity search through [Vicinity](https://github.com/MinishLab/vicinity).

SemHash supports both single-dataset deduplication (e.g., cleaning up a train set) and multi-dataset deduplication (e.g., ensuring no overlap between a test set and a train set). It works with simple datasets, such as text lists, and more complex ones, like multi-column QA datasets. Additionally, it includes functions to inspect deduplication results, making it easier to understand and refine your data cleaning process.

## Quickstart

Install the package with:
```bash
pip install semhash
```

Deduplicate a single dataset with the following code (note: the examples assume you have `datasets` installed, which you can install with `pip install datasets`):

```python
from datasets import load_dataset
from semhash import SemHash

# Load a dataset to deduplicate
texts = load_dataset("ag_news", split="train")["text"]

# Initialize a SemHash instance
semhash = SemHash.from_records(records=texts)

# Deduplicate the texts
deduplicated_texts = semhash.self_deduplicate().deduplicated
```

Or, deduplicate across two datasets with the following code (e.g., eliminating train/test leakage):

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

Or, deduplicate multi-column datasets with the following code (e.g., deduplicating a QA dataset):

```python
from datasets import load_dataset
from semhash import SemHash

# Load the dataset
dataset = load_dataset("squad_v2", split="train")

# Convert the dataset to a list of dictionaries
records = [dict(row) for row in dataset]

# Initialize SemHash with the columns to deduplicate
semhash = SemHash.from_records(records=records, columns=["question", "context"])

# Deduplicate the records
deduplicated_records = semhash.self_deduplicate().deduplicated
```

The `deduplicate` and `self_deduplicate` functions return a [DeduplicationResult](https://github.com/MinishLab/semhash/blob/prepare-release/semhash/datamodels.py#L35). This object stores the deduplicated corpus, a set of duplicate objec (along with the objects that caused duplication), and several useful functions to further inspect the deduplication result. Examples of how these functions can be used can be found in the [usage](#usage) section.

## Main Features

- **Fast**: SemHash uses [model2vec](https://github.com/MinishLab/model2vec) to embed texts and [vicinity](https://github.com/MinishLab/vicinity) to perform similarity search, making it extremely fast.
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
deduplicated_texts = semhash.self_deduplicate()
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
records = [dict(row) for row in dataset]

# Initialize SemHash with the columns to deduplicate
semhash = SemHash.from_records(records=records, columns=["question", "context"])

# Deduplicate the records
deduplicated_records = semhash.self_deduplicate().deduplicated
```

</details>

<details>
<summary>  DeduplicationResult functionality </summary>
<br>

The `DeduplicationResult` object returned by the `deduplicate` and `self_deduplicate` functions contains several useful functions to inspect the deduplication result. The following code snippet shows how to use these functions:

```python
from datasets import load_dataset
from semhash import SemHash

# Load a dataset to deduplicate
texts = load_dataset("ag_news", split="train")["text"]

# Initialize a SemHash instance
semhash = SemHash.from_records(records=texts)

# Deduplicate the texts
deduplication_result = semhash.self_deduplicate()

# Check the texts
deduplication_result.deduplicated
# Check the duplicates
deduplication_result.duplicates
# See how many texts were duplicates
deduplication_result.duplicate_ratio
# See how many were exact duplicates
deduplication_result.exact_duplicate_ratio

# Get the least similar text from the duplicates. This is useful for finding the right threshold for deduplication.
least_similar = deduplication_result.get_least_similar_from_duplicates()

# Rethreshold the duplicates. This allows you to instantly rethreshold the duplicates with a new threshold without having to re-deduplicate the texts.
deduplication_result.rethreshold(0.95)
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
deduplicated_texts = semhash.self_deduplicate()
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
deduplicated_texts = semhash.self_deduplicate()
```

</details>

NOTE: By default, we use the ANN (approximate-nearest neighbors) backend for deduplication. We recommend keeping this since the recall for smaller datasets is ~100%, and it's needed for larger datasets (>1M samples) since these will take too long to deduplicate without ANN. If you want to use the flat/exact-matching backend, you can set `use_ann=False` in the SemHash constructor:

```python
semhash = SemHash.from_records(records=texts, use_ann=False)
```

## Benchmarks

We've benchmarked SemHash on a variety of datasets to measure the deduplication performance and speed. The benchmarks were run with the following setup:
- The benchmarks were all run on CPU
- The benchmarks were all run with `use_ann=True`
- The used encoder is the default encoder ([potion-base-8M](https://huggingface.co/minishlab/potion-base-8M)).
- The timings include the encoding time, index building time, and deduplication time.

### Train Deduplication Benchmark

| Dataset               |      Original Train Size |      Deduplicated Train Size |    % Removed | Deduplication Time (s) |
|-----------------------|-------------------------:|-----------------------------:|-------------:|-----------------------:|
| bbc                   |                     1225 |                        1144 |         6.61 |                  0.27 |
| senteval_cr           |                     3012 |                        2990 |         0.73 |                  0.13 |
| tweet_sentiment_extraction |               27481 |                       26695 |         2.86 |                  1.65 |
| emotion               |                    16000 |                       15695 |         1.91 |                  0.69 |
| amazon_counterfactual |                     5000 |                        4992 |         0.16 |                  0.33 |
| ag_news               |                   120000 |                      106926 |        10.89 |                  4.73 |
| enron_spam            |                    31716 |                       20542 |        35.23 |                  1.90 |
| subj                  |                     8000 |                        7990 |         0.12 |                  0.60 |
| sst5                  |                     8544 |                        8526 |         0.21 |                  0.59 |
| 20_newgroups          |                    11314 |                       10672 |         5.67 |                  0.70 |
| hatespeech_offensive  |                    22783 |                       22090 |         3.04 |                  0.92 |
| ade                   |                    17637 |                       15718 |        10.88 |                  0.73 |
| imdb                  |                    25000 |                       24831 |         0.68 |                  1.75 |
| massive_scenario      |                    11514 |                        9366 |        18.66 |                  0.47 |
| student               |                   117519 |                       63852 |        45.67 |                  8.22 |
| squad_v2              |                   130319 |                      115546 |        11.34 |                 11.76 |
| wikitext              |                  1801350 |                      884626 |        50.89 |                 73.29 |

### Train/Test Deduplication Benchmark

| Dataset               |      Train Size |      Test Size |      Deduplicated Test Size |    % Removed | Deduplication Time (s) |
|-----------------------|----------------:|---------------:|---------------------------:|-------------:|-----------------------:|
| bbc                   |            1225 |           1000 |                        872 |        12.80 |                  0.44 |
| senteval_cr           |            3012 |            753 |                        750 |         0.40 |                  0.12 |
| tweet_sentiment_extraction |      27481 |           3534 |                       3412 |         3.45 |                  1.42 |
| emotion               |           16000 |           2000 |                       1926 |         3.70 |                  0.60 |
| amazon_counterfactual |            5000 |           5000 |                       4990 |         0.20 |                  0.52 |
| ag_news               |          120000 |           7600 |                       6198 |        18.45 |                  3.62 |
| enron_spam            |           31716 |           2000 |                       1060 |        47.00 |                  1.96 |
| subj                  |            8000 |           2000 |                       1999 |         0.05 |                  0.63 |
| sst5                  |            8544 |           2210 |                       2205 |         0.23 |                  0.60 |
| 20_newgroups          |           11314 |           7532 |                       7095 |         5.80 |                  2.28 |
| hatespeech_offensive  |           22783 |           2000 |                       1925 |         3.75 |                  0.75 |
| ade                   |           17637 |           5879 |                       4953 |        15.75 |                  0.82 |
| imdb                  |           25000 |          25000 |                      24796 |         0.82 |                  2.76 |
| massive_scenario      |           11514 |           2974 |                       2190 |        26.36 |                  0.48 |
| student               |          117519 |           5000 |                       2398 |        52.04 |                  3.54 |
| squad_v2              |          130319 |          11873 |                      11869 |         0.03 |                  9.51 |
| wikitext              |         1801350 |           4358 |                       2135 |        51.01 |                 39.21 |


As can be seen, SemHash is extremely fast, and scales to large datasets with millions of records. There are some notable examples of train/test leakage, such as `enron_spam` and `student`, where the test dataset contains a significant amount of semantic overlap with the training dataset.

### Reproducing the Benchmarks

To run the benchmarks yourself, you can use the following command:

```bash
python -m benchmarks.run_benchmarks
```
Optionally, the datasets can be updated in the [datasets.py](https://github.com/MinishLab/semhash/blob/main/benchmarks/datasets.py) file.
