

<h2 align="center">
  <img width="30%" alt="SemHash logo" src="assets/images/semhash_logo_v2.png"><br/>
  Fast Multimodal Semantic Deduplication & Filtering
</h2>


<div align="center">




<h2>
    <a href="https://pypi.org/project/semhash/"><img src="https://img.shields.io/pypi/v/semhash?color=%23007ec6&label=pypi%20package" alt="Package version"></a>
    <a href="https://pypi.org/project/semhash/"><img src="https://img.shields.io/pypi/pyversions/semhash" alt="Supported Python versions"></a>
    <a href="https://pepy.tech/project/semhash">
      <img src="https://static.pepy.tech/badge/semhash" alt="Downloads">
    </a>
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


SemHash is a lightweight, multimodal library for semantic deduplication, outlier filtering, and representative sample selection. Text works out of the box with fast [Model2Vec](https://github.com/MinishLab/model2vec) embeddings, and images, audio, and other modalities are supported with custom encoders.

SemHash supports both single-dataset operations (clean a training set) and cross-dataset operations (deduplicate test against train). It works with simple lists and complex multi-column datasets, and includes inspection tools to help you understand and refine results. All operations use [Vicinity](https://github.com/MinishLab/vicinity) for efficient similarity search.

## Quickstart

Install the package with:
```bash
pip install semhash
```

### Text Deduplication, Filtering & Representative Sampling

Deduplicate a single dataset, filter outliers, and find representative samples with the following code (note: the examples assume you have `datasets` installed, which you can install with `pip install datasets`):

```python
from datasets import load_dataset
from semhash import SemHash

# Load a dataset to deduplicate
texts = load_dataset("ag_news", split="train")["text"]

# Initialize a SemHash instance
semhash = SemHash.from_records(records=texts)

# Deduplicate the texts
deduplicated_texts = semhash.self_deduplicate().selected

# Filter outliers
filtered_texts = semhash.self_filter_outliers().selected

# Find representative texts
representative_texts = semhash.self_find_representative().selected
```

### Image Deduplication, Filtering & Representative Sampling

Deduplicate an image dataset, filter outliers, and find representative samples using a vision model (requires `pip install sentence-transformers`):

```python
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from semhash import SemHash

# Load an image dataset and vision model
model = SentenceTransformer('clip-ViT-B-32')
dataset = load_dataset("uoft-cs/cifar10", split="test")

# Initialize a SemHash instance with the 'img' column
semhash = SemHash.from_records(list(dataset), columns=["img"], model=model)

# Deduplicate the images
deduplicated_images = semhash.self_deduplicate().selected

# Filter outliers
filtered_images = semhash.self_filter_outliers().selected

# Find representative images
representative_images = semhash.self_find_representative().selected
```

### Cross-Dataset Deduplication, Filtering & Representative Sampling

Deduplicate across two datasets, filter outliers, and find representative samples (e.g., eliminating train/test leakage):

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

# Filter outliers from the test data against the training data,
# optionally with a specific percentage
filtered_test_texts = semhash.filter_outliers(records=test_texts, outlier_percentage=0.1).selected

# Find representative texts in the test data against the training data,
# optionally with a specific selection size
representative_test_texts = semhash.find_representative(records=test_texts, selection_size=10).selected
```

### Multi-Column Deduplication

Deduplicate multi-column datasets (e.g., deduplicating a QA dataset):

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

The `deduplicate` and `self_deduplicate` functions return a [DeduplicationResult](https://github.com/MinishLab/semhash/blob/main/semhash/datamodels.py#L58). This object stores the deduplicated corpus, a set of duplicate objects (along with the objects that caused duplication), and several useful functions to further inspect the deduplication result.

The `filter_outliers`, `self_filter_outliers`, `find_representative`, and `self_find_representative` functions return a [FilterResult](https://github.com/MinishLab/semhash/blob/main/semhash/datamodels.py#179). This object stores the found outliers/representative samples.

For both the `DeduplicationResult` and `FilterResult` objects, you can easily view the filtered records with the `selected` attribute (e.g. to view outliers: `outliers = semhash.self_filter_outliers().filtered`)

### Inspecting Deduplication Results

The `DeduplicationResult` object provides powerful tools for understanding and refining your deduplication:

```python
from datasets import load_dataset
from semhash import SemHash

# Load and deduplicate a dataset
texts = load_dataset("ag_news", split="train")["text"]
semhash = SemHash.from_records(records=texts)
result = semhash.self_deduplicate()

# Access deduplicated and duplicate records
deduplicated_texts = result.selected
duplicate_texts = result.filtered

# View deduplication statistics
print(f"Duplicate ratio: {result.duplicate_ratio}")
print(f"Exact duplicate ratio: {result.exact_duplicate_ratio}")

# Find edge cases to tune your threshold
least_similar = result.get_least_similar_from_duplicates(n=5)

# Adjust threshold without re-deduplicating
result.rethreshold(0.95)

# View each kept record with its duplicate cluster
for item in result.selected_with_duplicates:
    print(f"Kept: {item.record}")
    print(f"Duplicates: {item.duplicates}")  # List of (duplicate_text, similarity_score)
```

## Main Features

- **Fast**: SemHash uses [model2vec](https://github.com/MinishLab/model2vec) to embed texts and [vicinity](https://github.com/MinishLab/vicinity) to perform similarity search, making it extremely fast.
- **Scalable**: SemHash can deduplicate & filter large datasets with millions of records thanks to the ANN backends in Vicinity.
- **Flexible**: SemHash can be used to deduplicate & filter a single dataset or across two datasets, and can also be used to deduplicate & filter multi-column datasets (such as QA datasets).
- **Lightweight**: SemHash is a lightweight package with minimal dependencies, making it easy to install and use.
- **Explainable**: Easily inspect the duplicates and what caused them with the `DeduplicationResult` object. You can also view the lowest similarity duplicates to find the right threshold for deduplication for your dataset.

## Usage

The following examples show the various ways you can use SemHash to deduplicate datasets, filter outliers, and find representative samples. These examples assume you have the `datasets` library installed, which you can install with `pip install datasets`.

<details>
<summary>  Deduplicate, filter outliers, and find representative samples on a single dataset </summary>
<br>

The following code snippet shows how to deduplicate a single dataset, filter outliers, and find representative samples using SemHash (in this example, the train split of the [AG News dataset](https://huggingface.co/datasets/fancyzhx/ag_news)):

```python
from datasets import load_dataset
from semhash import SemHash

# Load a dataset to deduplicate
texts = load_dataset("ag_news", split="train")["text"]

# Initialize a SemHash instance
semhash = SemHash.from_records(records=texts)

# Deduplicate the texts
deduplicated_texts = semhash.self_deduplicate().selected

# Filter outliers
filtered_texts = semhash.self_filter_outliers().selected

# Find representative texts
representative_texts = semhash.self_find_representative().selected
```
</details>

<details>
<summary>  Deduplicate, filter outliers, and find representative samples across two datasets </summary>
<br>

The following code snippet shows how to deduplicate across two datasets, filter outliers, and find representative samples using SemHash (in this example, the train/test split of the [AG News dataset](https://huggingface.co/datasets/fancyzhx/ag_news)):

```python
from datasets import load_dataset
from semhash import SemHash

# Load two datasets to deduplicate
train_texts = load_dataset("ag_news", split="train")["text"]
test_texts = load_dataset("ag_news", split="test")["text"]

# Initialize a SemHash instance with the training data
semhash = SemHash.from_records(records=train_texts)

# Deduplicate the test data against the training data
deduplicated_test_texts = semhash.deduplicate(records=test_texts).selected

# Filter outliers from the test data
filtered_test_texts = semhash.filter_outliers(records=test_texts).selected

# Find representative texts in the test data
representative_test_texts = semhash.find_representative(records=test_texts).selected
```

</details>

<details>
<summary>  Deduplicate, filter outliers, and find representative samples on multi-column datasets </summary>
<br>

The following code snippet shows how to deduplicate multi-column datasets, filter outliers, and find representative samples using SemHash (in this example, the train split of the QA dataset [SQuAD 2.0](https://huggingface.co/datasets/rajpurkar/squad_v2), which consists of questions, contexts, and answers):

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

# Filter outliers from the records
filtered_records = semhash.self_filter_outliers().selected

# Find representative samples in the records
representative_records = semhash.self_find_representative().selected
```

</details>

<details>
<summary>  Deduplicate, filter outliers, and find representative samples on image datasets </summary>
<br>

You can bring your own encoder for any modality by implementing the Encoder protocol. Here's an example using a vision model from timm for image deduplication:

```python
from datasets import load_dataset
import timm
import torch
from semhash import SemHash

# Requires: pip install timm torch datasets

# Create a custom image encoder
class VisionEncoder:
    """Custom encoder using timm models. Implements the Encoder protocol."""

    def __init__(self, model_name: str = "mobilenetv3_small_100.lamb_in1k"):
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0).eval()
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)

    def encode(self, inputs, batch_size: int = 128):
        """Encode a batch of PIL images into embeddings."""
        import numpy as np

        # Convert grayscale to RGB if needed
        rgb_inputs = [img.convert("RGB") if img.mode != "RGB" else img for img in inputs]

        # Process in batches to avoid memory issues
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(rgb_inputs), batch_size):
                batch_inputs = rgb_inputs[i : i + batch_size]
                batch = torch.stack([self.transform(img) for img in batch_inputs])
                embeddings = self.model(batch).numpy()
                all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

# Load image dataset
dataset = load_dataset("uoft-cs/cifar10", split="test")
train_data = [{"img": img, "id": i} for i, img in enumerate(dataset["img"][:100])]
test_data = [{"img": img, "id": i} for i, img in enumerate(dataset["img"][100:150])]

# Initialize SemHash with the custom vision encoder
semhash = SemHash.from_records(train_data, columns=["img"], model=VisionEncoder())

# Single-dataset operations
deduplicated = semhash.self_deduplicate().selected
outliers = semhash.self_filter_outliers().selected
representatives = semhash.self_find_representative().selected

# Cross-dataset operations
test_deduplicated = semhash.deduplicate(test_data).selected
test_outliers = semhash.filter_outliers(test_data).selected
test_representatives = semhash.find_representative(test_data, selection_size=10).selected
```

The Encoder protocol requires only an `encode(inputs, **kwargs)` method that returns a numpy array. This makes it easy to integrate any embedding model for any modality.

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
model = StaticModel.from_pretrained("minishlab/potion-multilingual-128M")

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
<summary>  Using custom ANN backends </summary>
<br>

By default, we use [USearch](https://github.com/unum-cloud/USearch) as the ANN (approximate-nearest neighbors) backend for deduplication. We recommend keeping this since the recall for smaller datasets is ~100%, and it's needed for larger datasets (>1M samples) since these will take too long to deduplicate without ANN. If you want to use a flat/exact-matching backend, you can set `ann_backend=Backend.BASIC` in the SemHash constructor:

```python
from semhash import SemHash
from vicinity import Backend

semhash = SemHash.from_records(records=texts, ann_backend=Backend.BASIC)
```

Any backend from [Vicinity](https://github.com/MinishLab/vicinity) can be used with SemHash. The following code snippet shows how to use [FAISS](https://github.com/facebookresearch/faiss) with a custom `nlist` parameter:

```python
from datasets import load_dataset
from semhash import SemHash
from vicinity import Backend

semhash = SemHash.from_records(records=texts, ann_backend=Backend.FAISS, nlist=50)
```

For the full list of supported ANN backends and args, see the [Vicinity docs](https://github.com/MinishLab/vicinity/tree/main?tab=readme-ov-file#supported-backends).

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
<summary> Initializing from embeddings </summary>
<br>
You can also initialize SemHash from pre-computed embeddings. The following code snippet shows how to do this:

```python
from datasets import load_dataset
from model2vec import StaticModel
from semhash import SemHash

# Load a dataset
texts = load_dataset("ag_news", split="train")["text"]

# Load an embedding model
model = StaticModel.from_pretrained("minishlab/potion-base-8M")

# Create embeddings
embeddings = model.encode(texts)

# Initialize SemHash from embeddings
semhash = SemHash.from_embeddings(embeddings=embeddings, records=texts, model=model)

# Deduplicate, filter outliers, and find representative samples
deduplicated_texts = semhash.self_deduplicate().selected
filtered_texts = semhash.self_filter_outliers().selected
representative_texts = semhash.self_find_representative().selected
```
</details>

<details>
<summary> Initializing from a HuggingFace Dataset </summary>
<br>
You can easily use SemHash with HuggingFace Datasets by converting them to a list:

```python
from datasets import load_dataset
from semhash import SemHash

# Load a HuggingFace dataset
dataset = load_dataset("ag_news", split="train")

# Convert to list and initialize SemHash
semhash = SemHash.from_records(records=list(dataset), columns=["text"])

# Deduplicate, filter outliers, and find representative samples
deduplicated_texts = semhash.self_deduplicate().selected
filtered_texts = semhash.self_filter_outliers().selected
representative_texts = semhash.self_find_representative().selected
```

This also works with multi-column datasets:

```python
from datasets import load_dataset
from semhash import SemHash

# Load a multi-column dataset
dataset = load_dataset("squad_v2", split="train")

# Convert to list and initialize with multiple columns
semhash = SemHash.from_records(records=list(dataset), columns=["question", "context"])

# Deduplicate the records
deduplicated_records = semhash.self_deduplicate().selected
```
</details>




## Benchmarks

SemHash is extremely fast and scales to large datasets with millions of records. We've benchmarked both text and image deduplication across a variety of datasets. For example, deduplicating text 1.8M records takes only ~83 seconds on CPU.

For detailed benchmark results and analysis, see the [benchmarks directory](benchmarks/README.md).

### Running Benchmarks

```bash
# Run text benchmarks
make benchmark-text

# Run image benchmarks
make benchmark-image

# Run all benchmarks
make benchmark
```

## License

MIT

## Citing

If you use SemHash in your research, please cite the following:
```bibtex
@software{minishlab2025semhash,
  author       = {{van Dongen}, Thomas and Stephan Tulkens},
  title        = {SemHash: Fast Multimodal Semantic Deduplication \& Filtering},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17265942},
  url          = {https://github.com/MinishLab/semhash},
  license      = {MIT}
}
```
