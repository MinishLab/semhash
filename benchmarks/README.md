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
- **Encoder**: Default encoder ([blink-8M-int8](https://huggingface.co/minishlab/blink-8M-int8))
- **Timing**: Includes encoding time, index building time, and deduplication time
- **Dependencies**: Requires `datasets` package (`pip install datasets`)

### Results

### Train Deduplication Benchmark

This benchmark measures the performance of deduplicating within a single training dataset.

| Dataset              |  Original Train Size |  Deduplicated Train Size |  % Removed |   Deduplication Time (s) |
|----------------------|----------------------|--------------------------|------------|--------------------------|
| bbc                  |                 1225 |                     1160 |       5.31 |                     0.09 |
| senteval_cr          |                 3012 |                     3004 |       0.27 |                     0.08 |
| tweet_sentiment_extraction |                27481 |                    27046 |       1.58 |                     0.92 |
| emotion              |                16000 |                    15889 |       0.69 |                     0.47 |
| amazon_counterfactual |                 5000 |                     4998 |       0.04 |                     0.12 |
| ag_news              |               120000 |                   109946 |       8.38 |                     4.73 |
| enron_spam           |                31716 |                    21852 |      31.10 |                     0.81 |
| subj                 |                 8000 |                     7991 |       0.11 |                     0.25 |
| sst5                 |                 8544 |                     8531 |       0.15 |                     0.33 |
| 20_newgroups         |                11314 |                    10742 |       5.06 |                     0.27 |
| hatespeech_offensive |                22783 |                    22453 |       1.45 |                     0.61 |
| ade                  |                17637 |                    15798 |      10.43 |                     0.66 |
| imdb                 |                25000 |                    24875 |       0.50 |                     0.75 |
| massive_scenario     |                11514 |                    10468 |       9.08 |                     0.28 |
| student              |               117519 |                    91064 |      22.51 |                     4.34 |
| squad_v2             |               130319 |                   112866 |      13.39 |                     6.29 |
| wikitext             |              1801350 |                   935800 |      48.05 |                    67.29 |

### Train/Test Deduplication Benchmark

This benchmark measures the performance of deduplicating a test dataset against a training dataset (detecting train/test leakage).

| Dataset              |   Train Size |    Test Size |   Deduplicated Test Size |  % Removed |   Deduplication Time (s) |
|----------------------|--------------|--------------|--------------------------|------------|--------------------------|
| bbc                  |         1225 |         1000 |                      894 |      10.60 |                     0.11 |
| senteval_cr          |         3012 |          753 |                      751 |       0.27 |                     0.07 |
| tweet_sentiment_extraction |        27481 |         3534 |                     3453 |       2.29 |                     0.62 |
| emotion              |        16000 |         2000 |                     1968 |       1.60 |                     0.33 |
| amazon_counterfactual |         5000 |         5000 |                     4995 |       0.10 |                     0.17 |
| ag_news              |       120000 |         7600 |                     6419 |      15.54 |                     3.03 |
| enron_spam           |        31716 |         2000 |                     1115 |      44.25 |                     0.53 |
| subj                 |         8000 |         2000 |                     1999 |       0.05 |                     0.20 |
| sst5                 |         8544 |         2210 |                     2207 |       0.14 |                     0.26 |
| 20_newgroups         |        11314 |         7532 |                     7134 |       5.28 |                     0.33 |
| hatespeech_offensive |        22783 |         2000 |                     1951 |       2.45 |                     0.43 |
| ade                  |        17637 |         5879 |                     4989 |      15.14 |                     0.51 |
| imdb                 |        25000 |        25000 |                    24869 |       0.52 |                     1.00 |
| massive_scenario     |        11514 |         2974 |                     2476 |      16.75 |                     0.24 |
| student              |       117519 |         5000 |                     3290 |      34.20 |                     2.47 |
| squad_v2             |       130319 |        11873 |                    11868 |       0.04 |                     3.78 |
| wikitext             |      1801350 |         4358 |                     2225 |      48.94 |                    40.45 |

### Key Findings

SemHash is extremely fast and scales to large datasets with millions of records. Some notable findings include:

- **Speed**: Deduplication is fast even for large datasets (e.g., 1.8M records in ~67 seconds)
- **Train/Test Leakage**: Several datasets show significant train/test overlap:
  - `enron_spam`: 44% of test data overlaps with training data
  - `student`: 34% of test data overlaps with training data
  - `wikitext`: 49% of test data overlaps with training data

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
