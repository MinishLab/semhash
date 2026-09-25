import json
import logging
from time import perf_counter
from typing import Any

from datasets import load_dataset
from model2vec import StaticModel

from benchmarks.data import DATASET_DICT
from semhash import SemHash

# Set up logging
logger = logging.getLogger(__name__)

MODES = ["semantic", "lexical"]


def main() -> None:
    """Run the benchmarks."""
    # Prepare lists to hold benchmark results
    train_dedup_results = []
    train_test_dedup_results = []

    model = StaticModel.from_pretrained("minishlab/potion-base-8m")
    mode_kwargs: dict[str, dict[str, Any]] = {"semantic": {"model": model}, "lexical": {"mode": "lexical"}}

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

        train_result: dict[str, Any] = {"dataset": dataset_name, "original_train_size": len(train_records)}
        train_test_result: dict[str, Any] = {
            "dataset": dataset_name,
            "train_size": len(train_records),
            "test_size": len(test_records),
        }
        for mode in MODES:
            # Build the SemHash instance
            build_start = perf_counter()
            semhash = SemHash.from_records(records=train_records, columns=columns, **mode_kwargs[mode])
            build_time = perf_counter() - build_start

            # Time how long it takes to deduplicate the train set
            train_start = perf_counter()
            deduplicated_train = semhash.self_deduplicate()
            train_time = perf_counter() - train_start

            # Time how long it takes to deduplicate the test set
            test_start = perf_counter()
            deduplicated_test = semhash.deduplicate(records=test_records)
            test_time = perf_counter() - test_start

            train_result[mode] = {
                "deduplicated_train_size": len(deduplicated_train.selected),
                "percent_removed": deduplicated_train.duplicate_ratio * 100,
                "build_time_seconds": build_time,
                "deduplication_time_seconds": train_time,
                "time_seconds": build_time + train_time,
            }
            train_test_result[mode] = {
                "deduplicated_test_size": len(deduplicated_test.selected),
                "percent_removed": deduplicated_test.duplicate_ratio * 100,
                "build_time_seconds": build_time,
                "deduplication_time_seconds": test_time,
                "time_seconds": build_time + test_time,
            }
            logger.info(
                f"[{mode.upper()}] Dataset: {dataset_name}\n"
                f" - Train % Removed: {deduplicated_train.duplicate_ratio * 100:.2f}\n"
                f" - Test % Removed: {deduplicated_test.duplicate_ratio * 100:.2f}\n"
                f" - Train Time (seconds): {build_time + train_time:.2f}\n"
                f" - Test Time (seconds): {build_time + test_time:.2f}\n"
            )

        train_dedup_results.append(train_result)
        train_test_dedup_results.append(train_test_result)

    # Write the results to JSON files
    with open("benchmarks/results/train_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(train_dedup_results, f, ensure_ascii=False, indent=2)

    with open("benchmarks/results/train_test_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(train_test_dedup_results, f, ensure_ascii=False, indent=2)

    mode_header = " | ".join(f"{f'% Removed ({m})':>20}" for m in MODES) + " | "
    mode_header += " | ".join(f"{f'Time {m} (s)':>18}" for m in MODES)
    mode_divider = ("-" * 22 + "|") * len(MODES) + ("-" * 20 + "|") * len(MODES)

    def mode_cells(result: dict[str, Any]) -> str:
        """Format the % removed and time cells of a result for every mode."""
        cells = [f"{result[m]['percent_removed']:>20.2f}" for m in MODES]
        cells += [f"{result[m]['time_seconds']:>18.2f}" for m in MODES]
        return " | ".join(cells)

    # Print the train table
    print("### Train Deduplication Benchmark\n")  # noqa T201
    print(f"| {'Dataset':<26} | {'Train Size':>12} | {mode_header} |")  # noqa T201
    print("|" + "-" * 28 + "|" + "-" * 14 + "|" + mode_divider)  # noqa T201
    for r in train_dedup_results:
        print(f"| {r['dataset']:<26} | {r['original_train_size']:>12} | {mode_cells(r)} |")  # noqa T201

    print("\n")  # noqa T201

    # Print the train/test table
    print("### Train/Test Deduplication Benchmark\n")  # noqa T201
    print(f"| {'Dataset':<26} | {'Train Size':>12} | {'Test Size':>12} | {mode_header} |")  # noqa T201
    print("|" + "-" * 28 + "|" + "-" * 14 + "|" + "-" * 14 + "|" + mode_divider)  # noqa T201
    for r in train_test_dedup_results:
        print(  # noqa T201
            f"| {r['dataset']:<26} | {r['train_size']:>12} | {r['test_size']:>12} | {mode_cells(r)} |"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
