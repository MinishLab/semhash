from typing import Sequence

from frozendict import frozendict

from semhash.datamodels import DuplicateRecord


def to_frozendict(record: dict[str, str], columns: set[str]) -> frozendict[str, str]:
    """Convert a record to a frozendict."""
    return frozendict({k: record.get(k, "") for k in columns})


def unpack_record(record: dict[str, str], columns: Sequence[str]) -> str:
    r"""
    Unpack a record into a single string.

    Uses self.columns to determine the order of the text segments.
    Each text is cleaned by replacing '\t' with ' '. The texts are then joined by '\t'.

    :param record: A record to unpack.
    :param columns: Columns to unpack.
    :return: A single string representation of the record.
    """
    return "\t".join(record.get(c, "").replace("\t", " ") for c in columns)


def unpack_records(records: Sequence[dict[str, str]], columns: Sequence[str]) -> list[str]:
    """Unpack a list of records into a list of strings."""
    return [unpack_record(r, columns) for r in records]


def pack_record(record: str, columns: Sequence[str]) -> dict[str, str]:
    """
    Pack a record from a single string into a dictionary.

    :param record: A single string representation of the record.
    :param columns: Columns to pack.
    :return: A dictionary representation of the record.
    """
    return dict(zip(columns, record.split("\t")))


def pack_records(records: Sequence[str], columns: Sequence[str]) -> list[dict[str, str]]:
    """Pack a list of strings into a list of records."""
    return [dict(zip(columns, r.split("\t"))) for r in records]


def map_deduplication_result_to_strings(
    duplicates: list[DuplicateRecord], columns: Sequence[str]
) -> list[DuplicateRecord]:
    """Convert the record and duplicates in each DuplicateRecord back to strings if self.was_string is True."""
    mapped = []
    for dup_rec in duplicates:
        record_as_str = unpack_record(dup_rec.record, columns)
        duplicates_as_str = unpack_records(dup_rec.duplicates, columns)
        mapped.append(
            DuplicateRecord(
                record=record_as_str, duplicates=duplicates_as_str, exact=dup_rec.exact, scores=dup_rec.scores
            )
        )
    return mapped
