import json
import logging
from time import perf_counter

from datasets import load_dataset
from model2vec import StaticModel

from benchmarks.data import DATASET_DICT
from semhash import SemHash

# Set up logging
logger = logging.getLogger(__name__)


def main() -> None:  # noqa: C901
    """Run the benchmarks."""
    # Prepare lists to hold benchmark results
    train_dedup_results = []
    train_test_dedup_results = []
    # Load the model and initialize SemHash

    model = StaticModel.from_pretrained("minishlab/potion-base-8m")

    for dataset_name, record in DATASET_DICT.items():
        logger.info(f"Loading dataset: {dataset_name} from {record.name}")

        # Load train and test splits
        if record.sub_directory:
            train_ds = load_dataset(record.name, record.sub_directory, split=record.split_one)
            test_ds = load_dataset(record.name, record.sub_directory, split=record.split_two)
        else:
            train_ds = load_dataset(record.name, split=record.split_one)
            test_ds = load_dataset(record.name, split=record.split_two)

        # If the dataset has columns, use them
        if record.columns:
            columns = record.columns
            train_records = [dict(row) for row in train_ds]
            test_records = [dict(row) for row in test_ds]
        # Else, use the text_name
        else:
            train_records = train_ds[record.text_name]
            test_records = test_ds[record.text_name]
            columns = None

        # Build the SemHash instance
        build_start = perf_counter()
        semhash = SemHash.from_records(model=model, use_ann=True, records=train_records, columns=columns)
        build_end = perf_counter()
        build_time = build_end - build_start
        # Time how long it takes to deduplicate the train set
        train_only_start = perf_counter()
        deduplicated_train = semhash.self_deduplicate()
        train_only_end = perf_counter()

        train_only_dedup_time = train_only_end - train_only_start
        original_train_size = len(train_records)
        dedup_train_size = len(deduplicated_train.selected)

        percent_removed_train = deduplicated_train.duplicate_ratio * 100
        train_dedup_results.append(
            {
                "dataset": dataset_name,
                "original_train_size": original_train_size,
                "deduplicated_train_size": dedup_train_size,
                "percent_removed": percent_removed_train,
                "build_time_seconds": build_time,
                "deduplication_time_seconds": train_only_dedup_time,
                "time_seconds": train_only_dedup_time + build_time,
            }
        )

        logger.info(
            f"[TRAIN DEDUPLICATION] Dataset: {dataset_name}\n"
            f" - Original Train Size: {original_train_size}\n"
            f" - Deduplicated Train Size: {dedup_train_size}\n"
            f" - % Removed: {percent_removed_train:.2f}\n"
            f" - Deduplication Time (seconds): {train_only_dedup_time:.2f}\n"
            f" - Build Time (seconds): {build_time:.2f}\n"
            f" - Total Time (seconds): {train_only_dedup_time + build_time:.2f}\n"
        )

        # Time how long it takes to deduplicate the test set
        train_test_start = perf_counter()
        deduplicated_test = semhash.deduplicate(
            records=test_records,
        )
        train_test_end = perf_counter()
        train_test_dedup_time = train_test_end - train_test_start
        original_test_size = len(test_records)
        deduped_test_size = len(deduplicated_test.selected)
        percent_removed_test = deduplicated_test.duplicate_ratio * 100

        train_test_dedup_results.append(
            {
                "dataset": dataset_name,
                "train_size": original_train_size,
                "test_size": original_test_size,
                "deduplicated_test_size": deduped_test_size,
                "percent_removed": percent_removed_test,
                "build_time_seconds": build_time,
                "deduplication_time_seconds": train_test_dedup_time,
                "time_seconds": train_test_dedup_time + build_time,
            }
        )

        logger.info(
            f"[TRAIN/TEST DEDUPLICATION] Dataset: {dataset_name}\n"
            f" - Train Size: {original_train_size}\n"
            f" - Test Size: {original_test_size}\n"
            f" - Deduplicated Test Size: {deduped_test_size}\n"
            f" - % Removed: {percent_removed_test:.2f}\n"
            f" - Deduplication Time (seconds): {train_test_dedup_time:.2f}\n"
            f" - Build Time (seconds): {build_time:.2f}\n"
            f" - Total Time (seconds): {train_test_dedup_time + build_time:.2f}\n"
        )

    # Write the results to JSON files
    with open("benchmarks/results/train_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(train_dedup_results, f, ensure_ascii=False, indent=2)

    with open("benchmarks/results/train_test_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(train_test_dedup_results, f, ensure_ascii=False, indent=2)

    # Print the train table
    print("### Train Deduplication Benchmark\n")  # noqa T201
    print(  # noqa T201
        f"| {'Dataset':<20} | {'Original Train Size':>20} | {'Deduplicated Train Size':>24} | {'% Removed':>10} | {'Deduplication Time (s)':>24} |"
    )  # noqa T201
    print("|" + "-" * 22 + "|" + "-" * 22 + "|" + "-" * 26 + "|" + "-" * 12 + "|" + "-" * 26 + "|")  # noqa T201
    for r in train_dedup_results:
        print(  # noqa T201
            f"| {r['dataset']:<20} "
            f"| {r['original_train_size']:>20} "
            f"| {r['deduplicated_train_size']:>24} "
            f"| {r['percent_removed']:>10.2f} "
            f"| {r['time_seconds']:>24.2f} |"
        )

    print("\n")  # noqa T201

    # Print the train/test table
    print("### Train/Test Deduplication Benchmark\n")  # noqa T201
    print(  # noqa T201
        f"| {'Dataset':<20} | {'Train Size':>12} | {'Test Size':>12} | {'Deduplicated Test Size':>24} | {'% Removed':>10} | {'Deduplication Time (s)':>24} |"
    )  # noqa T201
    print("|" + "-" * 22 + "|" + "-" * 14 + "|" + "-" * 14 + "|" + "-" * 26 + "|" + "-" * 12 + "|" + "-" * 26 + "|")  # noqa T201
    for r in train_test_dedup_results:
        print(  # noqa T201
            f"| {r['dataset']:<20} "
            f"| {r['train_size']:>12} "
            f"| {r['test_size']:>12} "
            f"| {r['deduplicated_test_size']:>24} "
            f"| {r['percent_removed']:>10.2f} "
            f"| {r['time_seconds']:>24.2f} |"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
