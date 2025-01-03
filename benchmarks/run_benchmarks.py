import json
import logging
from time import perf_counter

from datasets import load_dataset
from model2vec import StaticModel

from benchmarks.datasets import DATASET_DICT
from semhash import SemHash

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the benchmark."""
    # Initialize the model and SemHash
    model = StaticModel.from_pretrained("minishlab/potion-base-8M")
    semhash = SemHash(model=model, ann=True)

    results = []

    # Iterate over each dataset
    for dataset_name, record in DATASET_DICT.items():
        logger.info(f"Loading dataset: {dataset_name} from {record.name}")

        # Handle sub_directory/config if needed
        if record.sub_directory:
            ds = load_dataset(record.name, record.sub_directory, split="train")
        else:
            ds = load_dataset(record.name, split="train")

        # If the dataset has columns, use them
        if record.columns:
            semhash.columns = record.columns
            records_to_deduplicate = []
            for row in ds:
                item = {}
                for col in record.columns:
                    # Convert to string if needed, or keep original
                    item[col] = str(row[col])
                records_to_deduplicate.append(item)
        else:
            # Single-column approach (just use text_name)
            records_to_deduplicate = ds[record.text_name]

        # Time the deduplication process
        start_time = perf_counter()
        deduplicated_records = semhash.fit_deduplicate(records=records_to_deduplicate)
        end_time = perf_counter()

        elapsed_time = end_time - start_time
        original_len = len(records_to_deduplicate)
        dedup_len = len(deduplicated_records)

        # Store and display results
        results.append(
            {
                "dataset": dataset_name,
                "original_size": original_len,
                "deduplicated_size": dedup_len,
                "time_seconds": elapsed_time,
            }
        )

        logger.info(
            f"Dataset: {dataset_name}\n"
            f" - Original Size: {original_len}\n"
            f" - Deduplicated Size: {dedup_len}\n"
            f" - Time (seconds): {elapsed_time:.2f}\n"
        )

    # Write results to a JSON file
    with open("benchmarks/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Also print them as a Markdown table
    print("## Benchmark Results\n")  # noqa T201
    print("| Dataset             | Original Size | Deduplicated Size | Time (seconds) |")  # noqa T201
    print("|---------------------|--------------:|-------------------:|---------------:|")  # noqa T201

    for r in results:
        print(  # noqa T201
            f"| {r['dataset']:<20} "
            f"| {r['original_size']:>13} "
            f"| {r['deduplicated_size']:>17} "
            f"| {r['time_seconds']:.2f}         |"
        )


if __name__ == "__main__":
    main()
