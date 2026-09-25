# SemHash Benchmarks

This directory contains the benchmarking code and results for SemHash. The benchmarks measure deduplication performance and speed across a variety of text and image datasets.

## Table of Contents

- [Text Benchmarks](#text-benchmarks)
  - [Setup](#setup)
  - [Results](#results)
  - [Key Findings](#key-findings)
  - [Running Text Benchmarks](#running-text-benchmarks)
- [Image Benchmarks](#image-benchmarks)
  - [Setup](#setup-1)
  - [Results](#results-1)
  - [Key Findings](#key-findings-1)
  - [Running Image Benchmarks](#running-image-benchmarks)
- [Lexical Benchmarks](#lexical-benchmarks)
  - [Setup](#setup-2)
  - [Results](#results-2)
  - [Key Findings](#key-findings-2)
  - [Running Lexical Benchmarks](#running-lexical-benchmarks)
- [Running All Benchmarks](#running-all-benchmarks)

## Text Benchmarks

### Setup

All text benchmarks were run with the following configuration:
- **CPU-only**: All benchmarks run on CPU (no GPU acceleration)
- **ANN backend**: Default backend (USearch)
- **Encoder**: Default encoder ([potion-base-8M](https://huggingface.co/minishlab/potion-base-8M)) for semantic mode, MinHash for lexical mode
- **Timing**: Includes encoding time, index building time, and deduplication time
- **Dependencies**: Requires `datasets` package (`pip install datasets`)

### Results

### Train Deduplication Benchmark

This benchmark measures the performance of deduplicating within a single training dataset.

| Dataset                    |   Train Size | % Removed (semantic) |  % Removed (lexical) |  Time semantic (s) |   Time lexical (s) |
|----------------------------|--------------|----------------------|----------------------|--------------------|--------------------|
| bbc                        |         1225 |                 6.61 |                 4.16 |               0.17 |               0.28 |
| senteval_cr                |         3012 |                 0.73 |                 0.00 |               0.10 |               0.16 |
| tweet_sentiment_extraction |        27481 |                 2.86 |                 0.24 |               1.84 |               2.32 |
| emotion                    |        16000 |                 1.91 |                 0.23 |               0.72 |               1.07 |
| amazon_counterfactual      |         5000 |                 0.16 |                 0.06 |               0.24 |               0.30 |
| ag_news                    |       120000 |                10.90 |                 2.54 |               6.45 |              13.92 |
| enron_spam                 |        31716 |                35.24 |                23.70 |               2.08 |               6.12 |
| subj                       |         8000 |                 0.12 |                 0.05 |               0.51 |               0.54 |
| sst5                       |         8544 |                 0.21 |                 0.12 |               0.47 |               0.55 |
| 20_newgroups               |        11314 |                 5.56 |                 3.29 |               0.65 |               1.98 |
| hatespeech_offensive       |        22783 |                 3.04 |                 0.57 |               1.01 |               2.02 |
| ade                        |        17637 |                10.88 |                 9.75 |               0.72 |               1.20 |
| imdb                       |        25000 |                 0.67 |                 0.48 |               1.70 |               4.56 |
| massive_scenario           |        11514 |                18.65 |                 3.03 |               0.45 |               0.65 |
| student                    |       117519 |                45.65 |                 1.59 |              10.70 |              12.03 |
| squad_v2                   |       130319 |                15.83 |                 9.28 |               9.95 |              20.84 |
| wikitext                   |      1801350 |                50.89 |                46.67 |              90.90 |             227.11 |

### Train/Test Deduplication Benchmark

This benchmark measures the performance of deduplicating a test dataset against a training dataset (detecting train/test leakage).

| Dataset                    |   Train Size |    Test Size | % Removed (semantic) |  % Removed (lexical) |  Time semantic (s) |   Time lexical (s) |
|----------------------------|--------------|--------------|----------------------|----------------------|--------------------|--------------------|
| bbc                        |         1225 |         1000 |                12.90 |                 7.20 |               0.28 |               0.47 |
| senteval_cr                |         3012 |          753 |                 0.40 |                 0.00 |               0.09 |               0.16 |
| tweet_sentiment_extraction |        27481 |         3534 |                 3.48 |                 0.40 |               1.56 |               1.76 |
| emotion                    |        16000 |         2000 |                 3.70 |                 0.70 |               0.63 |               0.81 |
| amazon_counterfactual      |         5000 |         5000 |                 0.20 |                 0.08 |               0.38 |               0.38 |
| ag_news                    |       120000 |         7600 |                18.45 |                 4.62 |               4.71 |               9.36 |
| enron_spam                 |        31716 |         2000 |                47.00 |                35.60 |               1.80 |               5.47 |
| subj                       |         8000 |         2000 |                 0.05 |                 0.00 |               0.52 |               0.43 |
| sst5                       |         8544 |         2210 |                 0.23 |                 0.09 |               0.47 |               0.46 |
| 20_newgroups               |        11314 |         7532 |                 5.75 |                 3.54 |               1.61 |               2.48 |
| hatespeech_offensive       |        22783 |         2000 |                 3.70 |                 0.75 |               0.80 |               1.27 |
| ade                        |        17637 |         5879 |                15.77 |                14.25 |               0.75 |               0.96 |
| imdb                       |        25000 |        25000 |                 0.82 |                 0.50 |               2.54 |               7.43 |
| massive_scenario           |        11514 |         2974 |                26.46 |                 5.41 |               0.43 |               0.52 |
| student                    |       117519 |         5000 |                52.08 |                 3.14 |               3.83 |               7.99 |
| squad_v2                   |       130319 |        11873 |                 0.08 |                 0.00 |               7.68 |              16.51 |
| wikitext                   |      1801350 |         4358 |                51.03 |                46.42 |              55.02 |             139.76 |

### Key Findings

SemHash is extremely fast and scales to large datasets with millions of records. Some notable findings include:

- **Speed**: Deduplication is fast even for large datasets (e.g., 1.8M records in ~91 seconds)
- **Lexical vs semantic**: Lexical mode removes fewer records, since it only finds duplicates that share wording. On `student` it removes 2% where semantic mode removes 46%. It is also 2 to 2.5x slower on the larger datasets
- **Train/Test Leakage**: Several datasets show significant train/test overlap:
  - `enron_spam`: 47% of test data overlaps with training data
  - `student`: 52% of test data overlaps with training data
  - `wikitext`: 51% of test data overlaps with training data

### Running Text Benchmarks

To run the text benchmarks yourself:

```bash
# Install dependencies
pip install datasets

# Run benchmarks
python -m benchmarks.run_text_benchmarks
# Or using make
make benchmark-text
```

## Image Benchmarks

### Setup

All image benchmarks were run with the following configuration:
- **Device**: Apple Silicon GPU (MPS)
- **ANN backend**: Default backend (USearch)
- **Encoder**: MobileNetV3-Small ([mobilenetv3_small_100.lamb_in1k](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k))
- **Batch size**: 128 images per batch
- **Timing**: Includes encoding time, index building time, and deduplication time

### Results

#### Train Deduplication Benchmark

This benchmark measures the performance of deduplicating within a single training dataset.

| Dataset              |  Original Train Size |  Deduplicated Train Size |  % Removed |   Deduplication Time (s) |
|----------------------|----------------------|--------------------------|------------|--------------------------|
| cifar10              |                50000 |                    48274 |       3.45 |                    61.20 |
| fashion_mnist        |                60000 |                    16714 |      72.14 |                    86.61 |

#### Train/Test Deduplication Benchmark

This benchmark measures the performance of deduplicating a test dataset against a training dataset.

| Dataset              |   Train Size |    Test Size |   Deduplicated Test Size |  % Removed |   Deduplication Time (s) |
|----------------------|--------------|--------------|--------------------------|------------|--------------------------|
| cifar10              |        50000 |        10000 |                     9397 |       6.03 |                    67.43 |
| fashion_mnist        |        60000 |        10000 |                     2052 |      79.48 |                    72.14 |

### Key Findings

- **Fashion-MNIST high deduplication**: Fashion-MNIST shows very high duplication rates (72% train, 79% test) due to the simple nature of the dataset (10 clothing categories with similar items)
- **CIFAR-10 moderate deduplication**: CIFAR-10 shows lower duplication (3.45% train, 6.03% test) as it contains more diverse natural images
- **Speed**: Image deduplication is fast even for large datasets (60k images in ~87 seconds on MPS); note that the actual deduplication step is quick, with most time spent on encoding images

### Running Image Benchmarks

To run the image benchmarks yourself:

```bash
# Install dependencies
pip install timm torch datasets

# Run benchmarks
python -m benchmarks.run_image_benchmarks
# Or using make
make benchmark-image
```

The image datasets can be customized by editing `benchmarks/data.py` (see `IMAGE_DATASET_DICT`).

## Lexical Benchmarks

### Setup

These benchmarks compare SemHash's lexical mode with [datasketch](https://github.com/ekzhu/datasketch) MinHashLSH:
- **Dataset**: `SetFit/ag_news`, where every document is paired with a copy that has 10% of its words replaced
- **Threshold**: Jaccard similarity of 0.7 over word 3-grams
- **Recall**: The fraction of copies with a true Jaccard similarity above the threshold that are found
- **Timing**: Includes encoding time, index building time, and deduplication time

### Results

| Implementation |  Records | Time (s) | Recall |
|----------------|----------|----------|--------|
| semhash        |    20000 |     1.76 |   0.91 |
| datasketch     |    20000 |     2.08 |   0.76 |
| semhash        |   100000 |     9.32 |   0.88 |
| datasketch     |   100000 |    11.25 |   0.76 |

### Key Findings

- **Higher recall**: SemHash finds around 89% of the duplicates, against 76% for datasketch, and is slightly faster
- **Smaller signatures**: SemHash stores 64 bytes per record, against 512 bytes for datasketch

### Running Lexical Benchmarks

To run the lexical benchmarks yourself:

```bash
# Install dependencies
pip install datasets datasketch

# Run benchmarks
python -m benchmarks.run_lexical_benchmarks
# Or using make
make benchmark-lexical
```

## Running All Benchmarks

To run the text, image and lexical benchmarks:

```bash
make benchmark
```
