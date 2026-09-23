import argparse
import json
import logging
import random
import resource
import subprocess
import sys
from time import perf_counter

import numpy as np
from datasets import load_dataset

from semhash import SemHash
from semhash.minhash import MinHashEncoder, unpack_signatures

logger = logging.getLogger(__name__)

DATASET = "SetFit/ag_news"
NGRAM = 3
THRESHOLD = 0.7
SEMHASH_PERM = 512
DATASKETCH_PERM = 128


def load_docs(n: int) -> list[str]:
    """Load n documents, repeating the dataset with a distinguishing suffix if n exceeds its size."""
    texts = load_dataset(DATASET, split="train")["text"]
    return [texts[i % len(texts)] + (f" tail{i // len(texts)}" if i >= len(texts) else "") for i in range(n)]


def shingles(text: str) -> set[str]:
    """Split a text into its set of word n-grams."""
    words = text.split()
    if len(words) < NGRAM:
        return {text}
    return {" ".join(words[i : i + NGRAM]) for i in range(len(words) - NGRAM + 1)}


def perturb(text: str, keep: float) -> str:
    """Replace a fraction of the words in a text with random tokens."""
    return " ".join(w if random.random() < keep else f"zz{random.randint(0, 9999)}" for w in text.split())


def peak_rss_mb() -> float:
    """Peak resident set size of this process in megabytes."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def bench_semhash(docs: list[str]) -> dict:
    """Encode, index and self-deduplicate with SemHash in lexical mode."""
    encoder = MinHashEncoder(num_perm=SEMHASH_PERM, ngram_size=NGRAM)
    encoder.encode(docs[:1])  # Warm up, so the first call does not pay for lazy imports.
    baseline = peak_rss_mb()

    start = perf_counter()
    signatures = encoder.encode(docs)
    encode_time = perf_counter() - start

    start = perf_counter()
    semhash = SemHash.from_embeddings(embeddings=signatures, records=docs, model=encoder)
    index_time = perf_counter() - start

    start = perf_counter()
    kept = len(semhash.self_deduplicate(threshold=THRESHOLD).selected)
    dedup_time = perf_counter() - start

    return {
        "encode_seconds": encode_time,
        "index_seconds": index_time,
        "dedup_seconds": dedup_time,
        "kept": kept,
        "signature_bytes_per_record": signatures.shape[1],
        "peak_rss_mb": peak_rss_mb() - baseline,
    }


def bench_datasketch(docs: list[str]) -> dict:
    """Encode, index and self-deduplicate with datasketch MinHashLSH."""
    from datasketch import LeanMinHash, MinHash, MinHashLSH

    baseline = peak_rss_mb()

    start = perf_counter()
    signatures = []
    for doc in docs:
        minhash = MinHash(num_perm=DATASKETCH_PERM)
        minhash.update_batch([s.encode("utf-8") for s in shingles(doc)])
        signatures.append(LeanMinHash(minhash))
    encode_time = perf_counter() - start

    start = perf_counter()
    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=DATASKETCH_PERM)
    with lsh.insertion_session() as session:
        for i, minhash in enumerate(signatures):
            session.insert(i, minhash)
    index_time = perf_counter() - start

    start = perf_counter()
    removed: set[int] = set()
    for i, minhash in enumerate(signatures):
        if i in removed:
            continue
        removed.update(j for j in lsh.query(minhash) if j != i)
    dedup_time = perf_counter() - start

    return {
        "encode_seconds": encode_time,
        "index_seconds": index_time,
        "dedup_seconds": dedup_time,
        "kept": len(docs) - len(removed),
        "signature_bytes_per_record": DATASKETCH_PERM * 4,
        "peak_rss_mb": peak_rss_mb() - baseline,
    }


BENCHMARKS = {"semhash": bench_semhash, "datasketch": bench_datasketch}


def run_accuracy(n_pairs: int) -> list[dict]:
    """Compare both estimators against exact Jaccard similarity on perturbed document pairs."""
    from datasketch import MinHash

    docs = load_docs(n_pairs)
    pairs = [(d, perturb(d, random.choice([0.3, 0.5, 0.7, 0.85, 0.95, 1.0]))) for d in docs]
    truth = np.array([len(a & b) / len(a | b) for a, b in ((shingles(x), shingles(y)) for x, y in pairs)])
    high = truth >= 0.5

    def report(name: str, estimate: np.ndarray, num_perm: int, nbytes: int) -> dict:
        error = estimate - truth
        return {
            "estimator": name,
            "num_perm": num_perm,
            "bytes_per_record": nbytes,
            "mae": float(np.abs(error).mean()),
            "mae_above_half": float(np.abs(error[high]).mean()),
            "bias": float(error.mean()),
        }

    results = []
    for num_perm in (64, 128, 256):

        def minhash(text: str, num_perm: int = num_perm) -> MinHash:
            m = MinHash(num_perm=num_perm)
            m.update_batch([s.encode("utf-8") for s in shingles(text)])
            return m

        estimate = np.array([minhash(a).jaccard(minhash(b)) for a, b in pairs])
        results.append(report("datasketch", estimate, num_perm, num_perm * 4))

    for num_perm in (256, 512, 1024):
        encoder = MinHashEncoder(num_perm=num_perm, ngram_size=NGRAM)
        left = unpack_signatures(encoder.encode([a for a, _ in pairs]))
        right = unpack_signatures(encoder.encode([b for _, b in pairs]))
        estimate = (left * right).sum(axis=1) / num_perm
        results.append(report("semhash", estimate, num_perm, num_perm // 8))

    return results


def run_recall(n: int) -> list[dict]:
    """Measure the fraction of known duplicate pairs each implementation actually finds."""
    from datasketch import LeanMinHash, MinHash, MinHashLSH

    base = load_docs(n // 2)
    docs: list[str] = []
    for doc in base:
        docs += [doc, perturb(doc, keep=0.9)]
    partner = {i: i ^ 1 for i in range(len(docs))}

    sets = [shingles(d) for d in docs]
    # Identical pairs are left out: SemHash removes exact duplicates before indexing, so they say nothing about
    # near-duplicate recall.
    truly_duplicate = {
        i
        for i in range(0, len(docs), 2)
        if docs[i] != docs[i + 1] and len(sets[i] & sets[i + 1]) / len(sets[i] | sets[i + 1]) >= THRESHOLD
    }
    targets = sorted(truly_duplicate | {i + 1 for i in truly_duplicate})

    encoder = MinHashEncoder(num_perm=SEMHASH_PERM, ngram_size=NGRAM)
    signatures = encoder.encode(docs)
    semhash = SemHash.from_embeddings(embeddings=signatures, records=docs, model=encoder)
    neighbours = semhash.index.query_threshold(signatures[targets], threshold=THRESHOLD)
    found = sum(any(docs[partner[i]] == rec["text"] for rec, _ in hits) for i, hits in zip(targets, neighbours))
    results = [{"implementation": "semhash", "n": n, "pairs": len(targets) // 2, "recall": found / len(targets)}]

    signatures = []
    for doc in docs:
        minhash = MinHash(num_perm=DATASKETCH_PERM)
        minhash.update_batch([s.encode("utf-8") for s in shingles(doc)])
        signatures.append(LeanMinHash(minhash))
    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=DATASKETCH_PERM)
    with lsh.insertion_session() as session:
        for i, minhash in enumerate(signatures):
            session.insert(i, minhash)
    found = sum(partner[i] in set(lsh.query(signatures[i])) for i in targets)
    results.append({"implementation": "datasketch", "n": n, "pairs": len(targets) // 2, "recall": found / len(targets)})

    return results


def run_scale_worker(implementation: str, n: int) -> None:
    """Run one scaling configuration and print it as JSON, so the parent can read its peak memory."""
    random.seed(0)
    result = {"implementation": implementation, "n": n, **BENCHMARKS[implementation](load_docs(n))}
    print(json.dumps(result))  # noqa: T201


def run_scale(scales: list[int]) -> list[dict]:
    """Run every scaling configuration in a subprocess, since peak memory is a per-process measurement."""
    results = []
    for n in scales:
        for implementation in BENCHMARKS:
            logger.info(f"Running {implementation} at n={n}")
            command = [
                sys.executable,
                "-m",
                "benchmarks.run_lexical_benchmarks",
                "--worker",
                implementation,
                "--n",
                str(n),
            ]
            output = subprocess.run(command, capture_output=True, text=True, check=True).stdout
            results.append(json.loads(output.strip().splitlines()[-1]))
    return results


def print_tables(accuracy: list[dict], scale: list[dict], recall: list[dict]) -> None:
    """Print all three result sets as markdown tables."""
    print("### Estimator Accuracy (vs exact Jaccard)\n")  # noqa: T201
    print(f"| {'Estimator':<12} | {'Permutations':>12} | {'Bytes/record':>12} | {'MAE':>8} | {'MAE (J>=0.5)':>12} |")  # noqa: T201
    print("|" + "-" * 14 + "|" + "-" * 14 + "|" + "-" * 14 + "|" + "-" * 10 + "|" + "-" * 14 + "|")  # noqa: T201
    for r in accuracy:
        print(  # noqa: T201
            f"| {r['estimator']:<12} | {r['num_perm']:>12} | {r['bytes_per_record']:>12} "
            f"| {r['mae']:>8.4f} | {r['mae_above_half']:>12.4f} |"
        )

    print("\n### Scaling\n")  # noqa: T201
    print(  # noqa: T201
        f"| {'Implementation':<14} | {'Records':>10} | {'Encode (s)':>10} | {'Index (s)':>10} "
        f"| {'Dedup (s)':>10} | {'Peak RSS (MB)':>14} |"
    )
    print("|" + "-" * 16 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 16 + "|")  # noqa: T201
    for r in scale:
        print(  # noqa: T201
            f"| {r['implementation']:<14} | {r['n']:>10} | {r['encode_seconds']:>10.2f} "
            f"| {r['index_seconds']:>10.2f} | {r['dedup_seconds']:>10.2f} | {r['peak_rss_mb']:>14.1f} |"
        )

    print("\n### Duplicate Pair Recall\n")  # noqa: T201
    print(f"| {'Implementation':<14} | {'Records':>10} | {'Known pairs':>12} | {'Recall':>8} |")  # noqa: T201
    print("|" + "-" * 16 + "|" + "-" * 12 + "|" + "-" * 14 + "|" + "-" * 10 + "|")  # noqa: T201
    for r in recall:
        print(  # noqa: T201
            f"| {r['implementation']:<14} | {r['n']:>10} | {r['pairs']:>12} | {r['recall']:>8.4f} |"
        )


def main() -> None:
    """Run the lexical deduplication benchmarks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", type=int, nargs="+", default=[20_000, 100_000])
    parser.add_argument("--recall-scales", type=int, nargs="+", default=[20_000, 100_000])
    parser.add_argument("--accuracy-pairs", type=int, default=1500)
    parser.add_argument("--worker", choices=list(BENCHMARKS))
    parser.add_argument("--n", type=int)
    args = parser.parse_args()

    if args.worker:
        run_scale_worker(args.worker, args.n)
        return

    random.seed(0)
    accuracy = run_accuracy(args.accuracy_pairs)
    scale = run_scale(args.scales)
    recall = [r for n in args.recall_scales for r in run_recall(n)]

    results = {"accuracy": accuracy, "scale": scale, "recall": recall}
    with open("benchmarks/results/lexical_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print_tables(accuracy, scale, recall)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
