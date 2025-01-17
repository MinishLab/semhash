from typing import Sequence

from frozendict import frozendict

from semhash.datamodels import DeduplicationResult, DuplicateRecord


def to_frozendict(record: dict[str, str], columns: set[str]) -> frozendict[str, str]:
    """Convert a record to a frozendict."""
    return frozendict({k: record.get(k, "") for k in columns})


def dict_to_string(record: dict[str, str], columns: Sequence[str]) -> str:
    r"""
    Turn a record into a single string.

    Uses self.columns to determine the order of the text segments.
    Each text is cleaned by replacing '\t' with ' '. The texts are then joined by '\t'.

    :param record: A record to unpack.
    :param columns: Columns to unpack.
    :return: A single string representation of the record.
    """
    return "\t".join(record.get(c, "").replace("\t", " ") for c in columns)


def map_deduplication_result_to_strings(result: DeduplicationResult, columns: Sequence[str]) -> DeduplicationResult:
    """Convert the record and duplicates in each DuplicateRecord back to strings if self.was_string is True."""
    deduplicated_str = [dict_to_string(r, columns) for r in result.deduplicated]
    mapped = []
    for dup_rec in result.duplicates:
        record_as_str = dict_to_string(dup_rec.record, columns)
        duplicates_as_str = [(dict_to_string(r, columns), score) for r, score in dup_rec.duplicates]
        mapped.append(
            DuplicateRecord(
                record=record_as_str,
                duplicates=duplicates_as_str,
                exact=dup_rec.exact,
            )
        )
    return DeduplicationResult(deduplicated=deduplicated_str, duplicates=mapped, threshold=result.threshold)


def add_scores_to_records(records: list[dict[str, str]]) -> list[tuple[dict[str, str], float]]:
    """Add scores to records and return a DeduplicationResult."""
    return [(record, 1.0) for record in records]
