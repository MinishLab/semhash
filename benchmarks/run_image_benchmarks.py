import json
import logging
from time import perf_counter

import numpy as np
import timm
import torch
from datasets import load_dataset

from benchmarks.data import IMAGE_DATASET_DICT
from semhash import SemHash

# Set up logging
logger = logging.getLogger(__name__)


class VisionEncoder:
    """Custom encoder using timm models for image embeddings."""

    def __init__(self, model_name: str = "mobilenetv3_small_100.lamb_in1k") -> None:
        """Initialize the vision encoder with a timm model."""
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        logger.info(f"Using device: {self.device}")

        self.model = timm.create_model(model_name, pretrained=True, num_classes=0).eval()
        self.model = self.model.to(self.device)

        data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)

    def encode(self, inputs: list, batch_size: int = 128) -> np.ndarray:
        """Encode a batch of PIL images into embeddings."""
        # Convert grayscale to RGB if needed
        rgb_inputs = [img.convert("RGB") if img.mode != "RGB" else img for img in inputs]

        # Process in batches
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(rgb_inputs), batch_size):
                batch_inputs = rgb_inputs[i : i + batch_size]
                batch = torch.stack([self.transform(img) for img in batch_inputs]).to(self.device)
                embeddings = self.model(batch).cpu().numpy()
                all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)


def main() -> None:  # noqa: C901
    """Run the image benchmarks."""
    # Prepare lists to hold benchmark results
    train_dedup_results = []
    train_test_dedup_results = []

    # Initialize vision encoder
    encoder = VisionEncoder()

    for dataset_name, record in IMAGE_DATASET_DICT.items():
        logger.info(f"Loading dataset: {dataset_name} from {record.name}")

        # Load train and test splits
        if record.sub_directory:
            train_ds = load_dataset(record.name, record.sub_directory, split=record.split_one)
            test_ds = load_dataset(record.name, record.sub_directory, split=record.split_two)
        else:
            train_ds = load_dataset(record.name, split=record.split_one)
            test_ds = load_dataset(record.name, split=record.split_two)

        # Convert to list of dicts with image column
        train_records = list(train_ds)
        test_records = list(test_ds)
        columns = record.columns

        # Build the SemHash instance
        build_start = perf_counter()
        semhash = SemHash.from_records(model=encoder, records=train_records, columns=columns)
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
    with open("benchmarks/results/image_train_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(train_dedup_results, f, ensure_ascii=False, indent=2)

    with open("benchmarks/results/image_train_test_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(train_test_dedup_results, f, ensure_ascii=False, indent=2)

    # Print the train table
    print("### Image Train Deduplication Benchmark\n")  # noqa T201
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
    print("### Image Train/Test Deduplication Benchmark\n")  # noqa T201
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
