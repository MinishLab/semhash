

<div align="center">
    <picture>
      <img width="30%" alt="SemHash logo" src="assets/images/semhash_logo.png">
    </picture>
  </a>
</div>

<div align="center">

# Fast Semantic Text Deduplication


<h2>
    <a href="https://pypi.org/project/semhash/"><img src="https://img.shields.io/pypi/v/semhash?color=%23007ec6&label=pypi%20package" alt="Package version"></a>
    <a href="https://pypi.org/project/semhash/"><img src="https://img.shields.io/pypi/pyversions/semhash" alt="Supported Python versions"></a>
    <a href="https://app.codecov.io/gh/MinishLab/semhash">
        <img src="https://codecov.io/gh/MinishLab/semhash/graph/badge.svg?token=YPOD6HD0MG" alt="Codecov">
    </a>
    <a href="https://discord.gg/4BDPR5nmtK">
        <img src="https://img.shields.io/badge/Join-Discord-5865F2?logo=discord&logoColor=white" alt="Join Discord">
    </a>
    <a href="https://github.com/MinishLab/semhash/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/license-MIT-green" alt="License - MIT">
    </a>
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
deduplicated_texts = semhash.self_deduplicate().selected
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

# Deduplicate the test data against the training data, optionally with a specific threshold
deduplicated_test_texts = semhash.deduplicate(records=test_texts, threshold=0.9).selected
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
deduplicated_records = semhash.self_deduplicate().selected
```

The `deduplicate` and `self_deduplicate` functions return a [DeduplicationResult](https://github.com/MinishLab/semhash/blob/main/semhash/datamodels.py#L30). This object stores the deduplicated corpus, a set of duplicate object (along with the objects that caused duplication), and several useful functions to further inspect the deduplication result.

Besides deduplication, SemHash also supports filtering through the `filter_by_entropy` and `self_filter_by_entropy` functions. These functions filter records based on their average distance to the top-k nearest neighbors. This is useful for finding examplars of the dataset and selecting the best records.

Examples of how these functions can be used can be found in the [usage](#usage) section.

## Main Features

- **Fast**: SemHash uses [model2vec](https://github.com/MinishLab/model2vec) to embed texts and [vicinity](https://github.com/MinishLab/vicinity) to perform similarity search, making it extremely fast.
- **Scalable**: SemHash can deduplicate large datasets with millions of records thanks to the ANN backends in Vicinity.
- **Flexible**: SemHash can be used to deduplicate a single dataset or across two datasets, and can also be used to deduplicate multi-column datasets (such as QA datasets).
- **Lightweight**: SemHash is a lightweight package with minimal dependencies, making it easy to install and use.
- **Explainable**: Easily inspect the duplicates and what caused them with the `DeduplicationResult` object. You can also view the lowest similarity duplicates to find the right threshold for deduplication for your dataset.

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
deduplicated_records = semhash.self_deduplicate().selected
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

# Check the deduplicated texts
deduplication_result.selected
# Check the duplicates
deduplication_result.filtered
# See what percentage of the texts were duplicates
deduplication_result.duplicate_ratio
# See what percentage of the texts were exact duplicates
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

<details>
<summary>  Using Pandas DataFrames </summary>
<br>

You can easily use Pandas DataFrames with SemHash. The following code snippet shows how to deduplicate a Pandas DataFrame:

```python
import pandas as pd
from datasets import load_dataset
from semhash import SemHash

# Load a dataset as a pandas dataframe
dataframe = load_dataset("ag_news", split="train").to_pandas()

# Convert the dataframe to a list of dictionaries
dataframe = dataframe.to_dict(orient="records")

# Initialize a SemHash instance with the columns to deduplicate
semhash = SemHash.from_records(records=dataframe, columns=["text"])

# Deduplicate the texts
deduplicated_records = semhash.self_deduplicate().selected

# Convert the deduplicated records back to a pandas dataframe
deduplicated_dataframe = pd.DataFrame(deduplicated_records)
```

</details>

<details>
<summary>  Filter based on entropy score </summary>


Entropy is a measure of the diversity of a dataset. A higher entropy score means more diverse records, and a lower entropy score means more similar records. You can use `budget` to filter on a percentage or a number of records. Setting `descending=True` will filter on the most diverse records.

Filter on a single dataset.

```python
from datasets import load_dataset
from semhash import SemHash

# Load a dataset to deduplicate
texts = load_dataset("ag_news", split="train")["text"]

# Initialize a SemHash instance
semhash = SemHash.from_records(records=texts)

# Filter the texts based on entropy score
filtered_records = semhash.self_filter_by_entropy(budget=0.8).selected
```

Filter across two datasets. This can be useful for finding examplars of the dataset before or after deduplication.

```python
from datasets import load_dataset
from semhash import SemHash

# Load a dataset
texts = load_dataset("ag_news", split="train")["text"]

# Initialize a SemHash instance
semhash = SemHash.from_records(records=texts)

# Deduplicate the records
deduplicated_records = semhash.self_deduplicate().selected

# Filter the records based on entropy score
filtered_records = semhash.filter_by_entropy(
  records=deduplicated_records,
  budget=0.8,
  descending=True
).selected
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

| Dataset              |  Original Train Size |  Deduplicated Train Size |  % Removed |   Deduplication Time (s) |
|----------------------|----------------------|--------------------------|------------|--------------------------|
| bbc                  |                 1225 |                     1144 |       6.61 |                     0.57 |
| senteval_cr          |                 3012 |                     2990 |       0.73 |                     0.14 |
| tweet_sentiment_extraction |                27481 |                    26695 |       2.86 |                     1.77 |
| emotion              |                16000 |                    15695 |       1.91 |                     0.77 |
| amazon_counterfactual |                 5000 |                     4992 |       0.16 |                     0.33 |
| ag_news              |               120000 |                   106921 |      10.90 |                     5.20 |
| enron_spam           |                31716 |                    20540 |      35.24 |                     2.03 |
| subj                 |                 8000 |                     7990 |       0.12 |                     0.63 |
| sst5                 |                 8544 |                     8526 |       0.21 |                     0.58 |
| 20_newgroups         |                11314 |                    10684 |       5.57 |                     0.73 |
| hatespeech_offensive |                22783 |                    22090 |       3.04 |                     0.92 |
| ade                  |                17637 |                    15718 |      10.88 |                     0.73 |
| imdb                 |                25000 |                    24830 |       0.68 |                     1.76 |
| massive_scenario     |                11514 |                     9366 |      18.66 |                     0.47 |
| student              |               117519 |                    63856 |      45.66 |                     8.80 |
| squad_v2             |               130319 |                   109698 |      15.82 |                     8.81 |
| wikitext             |              1801350 |                   884645 |      50.89 |                    83.53 |

### Train/Test Deduplication Benchmark

| Dataset              |   Train Size |    Test Size |   Deduplicated Test Size |  % Removed |   Deduplication Time (s) |
|----------------------|--------------|--------------|--------------------------|------------|--------------------------|
| bbc                  |         1225 |         1000 |                      870 |      13.00 |                     0.71 |
| senteval_cr          |         3012 |          753 |                      750 |       0.40 |                     0.13 |
| tweet_sentiment_extraction |        27481 |         3534 |                     3412 |       3.45 |                     1.53 |
| emotion              |        16000 |         2000 |                     1926 |       3.70 |                     0.65 |
| amazon_counterfactual |         5000 |         5000 |                     4990 |       0.20 |                     0.51 |
| ag_news              |       120000 |         7600 |                     6198 |      18.45 |                     3.74 |
| enron_spam           |        31716 |         2000 |                     1060 |      47.00 |                     1.94 |
| subj                 |         8000 |         2000 |                     1999 |       0.05 |                     0.62 |
| sst5                 |         8544 |         2210 |                     2205 |       0.23 |                     0.59 |
| 20_newgroups         |        11314 |         7532 |                     7098 |       5.76 |                     2.25 |
| hatespeech_offensive |        22783 |         2000 |                     1925 |       3.75 |                     0.77 |
| ade                  |        17637 |         5879 |                     4952 |      15.77 |                     0.81 |
| imdb                 |        25000 |        25000 |                    24795 |       0.82 |                     2.81 |
| massive_scenario     |        11514 |         2974 |                     2190 |      26.36 |                     0.46 |
| student              |       117519 |         5000 |                     2393 |      52.14 |                     3.78 |
| squad_v2             |       130319 |        11873 |                    11863 |       0.08 |                     7.13 |
| wikitext             |      1801350 |         4358 |                     2139 |      50.92 |                    40.32 |

As can be seen, SemHash is extremely fast, and scales to large datasets with millions of records. There are some notable examples of train/test leakage, such as `enron_spam` and `student`, where the test dataset contains a significant amount of semantic overlap with the training dataset.

### Reproducing the Benchmarks

To run the benchmarks yourself, you can use the following command (assuming you have the `datasets` library installed):

```bash
python -m benchmarks.run_benchmarks
```
Optionally, the datasets can be updated in the [datasets.py](https://github.com/MinishLab/semhash/blob/main/benchmarks/datasets.py) file.

## License

MIT

## Citing

If you use SemHash in your research, please cite the following:

```bibtex
@software{minishlab2025semhash,
  author = {Thomas van Dongen and Stephan Tulkens},
  title = {SemHash: Fast Semantic Text Deduplication},
  year = {2025},
  url = {https://github.com/MinishLab/semhash}
}
```
