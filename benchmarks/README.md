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
- **Encoder**: Default encoder ([potion-base-8M](https://huggingface.co/minishlab/potion-base-8M))
- **Timing**: Includes encoding time, index building time, and deduplication time
- **Dependencies**: Requires `datasets` package (`pip install datasets`)

### Results

### Train Deduplication Benchmark

This benchmark measures the performance of deduplicating within a single training dataset.

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

This benchmark measures the performance of deduplicating a test dataset against a training dataset (detecting train/test leakage).

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

### Key Findings

SemHash is extremely fast and scales to large datasets with millions of records. Some notable findings include:

- **Speed**: Deduplication is fast even for large datasets (e.g., 1.8M records in ~83 seconds)
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

These benchmarks compare SemHash's lexical mode against [datasketch](https://github.com/ekzhu/datasketch), the
reference MinHash / MinHashLSH implementation for Python.

### Setup

Both implementations deduplicate `SetFit/ag_news` at a Jaccard threshold of `0.7` over word 3-grams. SemHash uses
512 permutations of 1-bit MinHash in a usearch Hamming index; datasketch uses 128 permutations of 32-bit MinHash
in a banded LSH index. Peak memory is measured per configuration in a separate process.

Recall is measured against known duplicate pairs: every document is paired with a copy in which 10% of the words
are replaced, and the pairs whose *exact* Jaccard similarity is above the threshold are the ones that should be
found. Pairs where no word was replaced are left out, since SemHash removes exact duplicates before indexing. The
recall therefore includes estimator error, not just index error.

### Results

| Estimator    | Permutations | Bytes/record |      MAE | MAE (J>=0.5) |
|--------------|--------------|--------------|----------|--------------|
| datasketch   |           64 |          256 |   0.0261 |       0.0221 |
| datasketch   |          128 |          512 |   0.0185 |       0.0152 |
| datasketch   |          256 |         1024 |   0.0135 |       0.0113 |
| semhash      |          256 |           32 |   0.0378 |       0.0155 |
| semhash      |          512 |           64 |   0.0257 |       0.0120 |
| semhash      |         1024 |          128 |   0.0175 |       0.0084 |

| Implementation |    Records | Encode (s) |  Index (s) |  Dedup (s) |  Peak RSS (MB) |
|----------------|------------|------------|------------|------------|----------------|
| semhash        |      20000 |       0.45 |       0.64 |       0.87 |           36.5 |
| datasketch     |      20000 |       1.73 |       0.24 |       0.11 |          103.1 |
| semhash        |     100000 |       2.23 |       3.15 |       3.88 |          237.0 |
| datasketch     |     100000 |       8.89 |       1.47 |       0.68 |          627.2 |

| Implementation |    Records |  Known pairs |   Recall |
|----------------|------------|--------------|----------|
| semhash        |      20000 |         2168 |   0.9066 |
| datasketch     |      20000 |         2168 |   0.7611 |
| semhash        |     100000 |        11097 |   0.8813 |
| datasketch     |     100000 |        11097 |   0.7564 |

### Key Findings

- **Higher recall.** SemHash finds around 89% of known duplicate pairs against datasketch's 76%. LSH recall is set
  by its banding curve, which is flat in dataset size but caps well below 1 because pairs near the threshold sit on
  the shallow part of the curve.
- **Smaller signatures.** 1-bit MinHash stores one bit per permutation instead of 32, so a signature that is as
  accurate as datasketch's in the deduplication regime is 8x smaller, and peak memory is roughly halved.
- **Faster encoding, slower indexing.** Encoding is about 4x faster, because the permutations are applied to all
  shingles at once in wrapping uint32 arithmetic rather than per shingle. Building and querying an HNSW index is slower than hashing into LSH bands, and grows
  slightly superlinearly, so the total is comparable rather than better.
- **Recall needs a wide search.** HNSW recall degrades with dataset size at usearch's default expansion (down to
  0.61 at a million records), which is why the binary index sets it to 512. That holds recall above 0.96 on
  estimator-achievable pairs from 200k to 1M records, at roughly 4x the build and query cost.

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

Scales and the number of accuracy pairs can be changed with `--scales`, `--recall-scales` and `--accuracy-pairs`.

## Running All Benchmarks

To run the text, image and lexical benchmarks:

```bash
make benchmark
```
