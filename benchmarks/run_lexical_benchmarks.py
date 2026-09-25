import json
import logging
import random
from time import perf_counter

from datasets import load_dataset
from datasketch import MinHash, MinHashLSH

from semhash import SemHash

logger = logging.getLogger(__name__)

THRESHOLD = 0.7
SIZES = [20_000, 100_000]


def shingles(text: str) -> set[str]:
    """Split a text into its set of word 3-grams, the same n-grams SemHash uses by default."""
    words = text.split()
    return {" ".join(words[i : i + 3]) for i in range(len(words) - 2)} or {text}


def make_pairs(n: int) -> tuple[list[str], set[tuple[str, str]]]:
    """Pair n / 2 documents with a copy that has 10% of its words replaced, and return the true duplicate pairs."""
    texts = load_dataset("SetFit/ag_news", split="train")["text"]
    docs: list[str] = []
    for i in range(n // 2):
        doc = texts[i % len(texts)] + (f" copy{i // len(texts)}" if i >= len(texts) else "")
        docs += [doc, " ".join(w if random.random() < 0.9 else f"zz{random.randint(0, 9999)}" for w in doc.split())]
    # Identical pairs are left out, since SemHash removes exact duplicates before indexing.
    pairs = set()
    for a, b in zip(docs[::2], docs[1::2]):
        x, y = shingles(a), shingles(b)
        if a != b and len(x & y) / len(x | y) >= THRESHOLD:
            pairs.add((a, b))
    return docs, pairs


def run_semhash(docs: list[str]) -> set[tuple[str, str]]:
    """Self-deduplicate with SemHash in lexical mode and return the duplicate pairs it found."""
    result = SemHash.from_records(records=docs, mode="lexical").self_deduplicate(threshold=THRESHOLD)
    return {(original, dup.record) for dup in result.filtered for original, _ in dup.duplicates}


def run_datasketch(docs: list[str]) -> set[tuple[str, str]]:
    """Self-deduplicate with datasketch MinHashLSH and return the duplicate pairs it found."""
    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=128)
    minhashes = []
    with lsh.insertion_session() as session:
        for i, doc in enumerate(docs):
            minhash = MinHash(num_perm=128)
            minhash.update_batch([s.encode("utf-8") for s in shingles(doc)])
            session.insert(i, minhash)
            minhashes.append(minhash)
    return {(docs[j], docs[i]) for i, minhash in enumerate(minhashes) for j in lsh.query(minhash) if j < i}


def main() -> None:
    """Run the lexical benchmarks."""
    results = []
    for n in SIZES:
        random.seed(0)
        docs, pairs = make_pairs(n)
        for name, run in [("semhash", run_semhash), ("datasketch", run_datasketch)]:
            logger.info(f"Running {name} on {n} records")
            start = perf_counter()
            found = run(docs)
            results.append(
                {
                    "implementation": name,
                    "records": n,
                    "time_seconds": perf_counter() - start,
                    "recall": len(pairs & found) / len(pairs),
                }
            )

    with open("benchmarks/results/lexical_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"| {'Implementation':<14} | {'Records':>8} | {'Time (s)':>8} | {'Recall':>6} |")  # noqa T201
    print("|" + "-" * 16 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 8 + "|")  # noqa T201
    for r in results:
        print(  # noqa T201
            f"| {r['implementation']:<14} | {r['records']:>8} | {r['time_seconds']:>8.2f} | {r['recall']:>6.2f} |"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
