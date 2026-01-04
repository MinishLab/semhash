# SemHash Benchmarks

This directory contains the benchmarking code and results for SemHash. The benchmarks measure deduplication performance and speed across a variety of datasets.

## Setup

All benchmarks were run with the following configuration:
- **CPU-only**: All benchmarks run on CPU (no GPU acceleration)
- **ANN backend**: Default backend (USearch)
- **Encoder**: Default encoder ([potion-base-8M](https://huggingface.co/minishlab/potion-base-8M))
- **Timing**: Includes encoding time, index building time, and deduplication time

## Results

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

## Key Findings

SemHash is extremely fast and scales to large datasets with millions of records. Some notable findings include:

- **Speed**: Deduplication is fast even for large datasets (e.g., 1.8M records in ~83 seconds)
- **Train/Test Leakage**: Several datasets show significant train/test overlap:
  - `enron_spam`: 47% of test data overlaps with training data
  - `student`: 52% of test data overlaps with training data
  - `wikitext`: 51% of test data overlaps with training data

## Running the Benchmarks

To run the benchmarks yourself:

```bash
python -m benchmarks.run_benchmarks
```

The datasets can be customized by editing `benchmarks/data.py`.
