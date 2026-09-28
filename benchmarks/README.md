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
- [Running All Benchmarks](#running-all-benchmarks)

## Text Benchmarks

### Setup

All text benchmarks were run with the following configuration:
- **Device**: MacBook Pro (Apple M5, 48 GB RAM)
- **CPU-only**: All benchmarks run on CPU (no GPU acceleration)
- **ANN backend**: Default backend (USearch)
- **Encoder**: Default encoder ([potion-base-8M](https://huggingface.co/minishlab/potion-base-8M))
- **Timing**: Includes encoding time, index building time, and deduplication time
- **Dependencies**: Requires `datasets` package (`pip install datasets`)

### Results

### Train Deduplication Benchmark

This benchmark measures the performance of deduplicating within a single training dataset.

| Dataset              |  Original Train Size |  Deduplicated Train Size |  % Removed |   Deduplication Time (s) |
|----------------------|----------------------|--------------------------|------------|--------------------------|
| bbc                  |                 1225 |                     1148 |       6.29 |                     0.19 |
| senteval_cr          |                 3012 |                     2992 |       0.66 |                     0.15 |
| tweet_sentiment_extraction |                27481 |                    26775 |       2.57 |                     1.67 |
| emotion              |                16000 |                    15739 |       1.63 |                     0.68 |
| amazon_counterfactual |                 5000 |                     4992 |       0.16 |                     0.27 |
| ag_news              |               120000 |                   107882 |      10.10 |                     5.75 |
| enron_spam           |                31716 |                    21121 |      33.41 |                     1.63 |
| subj                 |                 8000 |                     7990 |       0.12 |                     0.45 |
| sst5                 |                 8544 |                     8526 |       0.21 |                     0.46 |
| 20_newgroups         |                11314 |                    10717 |       5.28 |                     0.61 |
| hatespeech_offensive |                22783 |                    22233 |       2.41 |                     0.96 |
| ade                  |                17637 |                    15723 |      10.85 |                     0.74 |
| imdb                 |                25000 |                    24847 |       0.61 |                     1.64 |
| massive_scenario     |                11514 |                     9665 |      16.06 |                     0.45 |
| student              |               117519 |                    69696 |      40.69 |                     8.28 |
| squad_v2             |               130319 |                   110480 |      15.22 |                     9.66 |
| wikitext             |              1801350 |                   900554 |      50.01 |                    74.31 |

### Train/Test Deduplication Benchmark

This benchmark measures the performance of deduplicating a test dataset against a training dataset (detecting train/test leakage).

| Dataset              |   Train Size |    Test Size |   Deduplicated Test Size |  % Removed |   Deduplication Time (s) |
|----------------------|--------------|--------------|--------------------------|------------|--------------------------|
| bbc                  |         1225 |         1000 |                      874 |      12.60 |                     0.29 |
| senteval_cr          |         3012 |          753 |                      750 |       0.40 |                     0.15 |
| tweet_sentiment_extraction |        27481 |         3534 |                     3411 |       3.48 |                     1.48 |
| emotion              |        16000 |         2000 |                     1926 |       3.70 |                     0.58 |
| amazon_counterfactual |         5000 |         5000 |                     4990 |       0.20 |                     0.44 |
| ag_news              |       120000 |         7600 |                     6201 |      18.41 |                     3.95 |
| enron_spam           |        31716 |         2000 |                     1064 |      46.80 |                     1.56 |
| subj                 |         8000 |         2000 |                     1999 |       0.05 |                     0.44 |
| sst5                 |         8544 |         2210 |                     2205 |       0.23 |                     0.45 |
| 20_newgroups         |        11314 |         7532 |                     7098 |       5.76 |                     1.51 |
| hatespeech_offensive |        22783 |         2000 |                     1925 |       3.75 |                     0.78 |
| ade                  |        17637 |         5879 |                     4953 |      15.75 |                     0.79 |
| imdb                 |        25000 |        25000 |                    24797 |       0.81 |                     2.55 |
| massive_scenario     |        11514 |         2974 |                     2188 |      26.43 |                     0.47 |
| student              |       117519 |         5000 |                     2400 |      52.00 |                     4.36 |
| squad_v2             |       130319 |        11873 |                    11863 |       0.08 |                     7.07 |
| wikitext             |      1801350 |         4358 |                     2134 |      51.03 |                    46.29 |

### Key Findings

SemHash is extremely fast and scales to large datasets with millions of records. Some notable findings include:

- **Speed**: Deduplication is fast even for large datasets (e.g., 1.8M records in ~74 seconds)
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
- **Device**: MacBook Pro (Apple M5, 48 GB RAM), GPU via MPS
- **ANN backend**: Default backend (USearch)
- **Encoder**: MobileNetV3-Small ([mobilenetv3_small_100.lamb_in1k](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k))
- **Batch size**: 128 images per batch
- **Timing**: Includes encoding time, index building time, and deduplication time

### Results

#### Train Deduplication Benchmark

This benchmark measures the performance of deduplicating within a single training dataset.

| Dataset              |  Original Train Size |  Deduplicated Train Size |  % Removed |   Deduplication Time (s) |
|----------------------|----------------------|--------------------------|------------|--------------------------|
| cifar10              |                50000 |                    48390 |       3.22 |                    38.05 |
| fashion_mnist        |                60000 |                    23549 |      60.75 |                    42.70 |

#### Train/Test Deduplication Benchmark

This benchmark measures the performance of deduplicating a test dataset against a training dataset.

| Dataset              |   Train Size |    Test Size |   Deduplicated Test Size |  % Removed |   Deduplication Time (s) |
|----------------------|--------------|--------------|--------------------------|------------|--------------------------|
| cifar10              |        50000 |        10000 |                     9397 |       6.03 |                    43.35 |
| fashion_mnist        |        60000 |        10000 |                     2056 |      79.44 |                    46.89 |

### Key Findings

- **Fashion-MNIST high deduplication**: Fashion-MNIST shows very high duplication rates (61% train, 79% test) due to the simple nature of the dataset (10 clothing categories with similar items)
- **CIFAR-10 moderate deduplication**: CIFAR-10 shows lower duplication (3.22% train, 6.03% test) as it contains more diverse natural images
- **Speed**: Image deduplication is fast even for large datasets (60k images in ~43 seconds on MPS); note that the actual deduplication step is quick, with most time spent on encoding images

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

## Running All Benchmarks

To run both text and image benchmarks:

```bash
make benchmark
```
