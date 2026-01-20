from __future__ import annotations

from collections.abc import Sequence
from math import ceil
from typing import Any, Generic, Literal

import numpy as np
from frozendict import frozendict
from model2vec import StaticModel
from pyversity import Strategy, diversify
from vicinity import Backend

from semhash.datamodels import DeduplicationResult, DuplicateRecord, FilterResult
from semhash.index import Index
from semhash.records import (
    add_scores_to_records,
    group_records_by_key,
    map_deduplication_result_to_strings,
    prepare_records,
    remove_exact_duplicates,
)
from semhash.utils import (
    Encoder,
    Record,
    coerce_value,
    compute_candidate_limit,
    featurize,
    to_frozendict,
)


class SemHash(Generic[Record]):
    def __init__(self, index: Index, model: Encoder, columns: Sequence[str], was_string: bool) -> None:
        """
        Initialize SemHash.

        :param index: An index.
        :param model: A model to use for featurization.
        :param columns: Columns of the records.
        :param was_string: Whether the records were strings. Used for mapping back to strings.
        """
        self.index = index
        self.model = model
        self.columns = columns
        self._was_string = was_string
        self._ranking_cache: FilterResult | None = None

    @classmethod
    def from_records(
        cls,
        records: Sequence[Record],
        columns: Sequence[str] | None = None,
        model: Encoder | None = None,
        ann_backend: Backend | str = Backend.USEARCH,
        **kwargs: Any,
    ) -> SemHash:
        """
        Initialize a SemHash instance from records.

        Removes exact duplicates, featurizes the records, and fits a vicinity index.

        :param records: A list of records (strings or dictionaries).
        :param columns: Columns to featurize if records are dictionaries.
        :param model: (Optional) An Encoder model. If None, the default model is used (minishlab/potion-base-8M).
        :param ann_backend: (Optional) The ANN backend to use. Defaults to Backend.USEARCH.
        :param **kwargs: Any additional keyword arguments to pass to the Vicinity index.
        :return: A SemHash instance with a fitted vicinity index.
        """
        # Prepare and validate records
        dict_records, columns, was_string = prepare_records(records, columns)

        # If no model is provided, load the default model
        if model is None:  # pragma: no cover
            model = StaticModel.from_pretrained("minishlab/potion-base-8M")

        # Group by exact match, preserving first-occurrence order
        deduplicated_records, items = group_records_by_key(dict_records, columns)

        # Create embeddings for deduplicated records only
        embeddings = featurize(deduplicated_records, columns, model)

        index = Index.from_vectors_and_items(vectors=embeddings, items=items, backend_type=ann_backend, **kwargs)
        return cls(index=index, model=model, columns=columns, was_string=was_string)

    @classmethod
    def from_embeddings(
        cls,
        embeddings: np.ndarray,
        records: Sequence[Record],
        model: Encoder,
        columns: Sequence[str] | None = None,
        ann_backend: Backend | str = Backend.USEARCH,
        **kwargs: Any,
    ) -> SemHash:
        """
        Initialize a SemHash instance from pre-computed embeddings.

        Removes exact duplicates, featurizes the records, and fits a vicinity index.

        :param embeddings: Pre-computed embeddings as a numpy array of shape (n_records, embedding_dim).
        :param records: A list of records (strings or dictionaries) corresponding to the embeddings.
        :param model: The Encoder model used for creating the embeddings.
        :param columns: Columns to use if records are dictionaries. If None and records are strings,
            defaults to ["text"].
        :param ann_backend: (Optional) The ANN backend to use. Defaults to Backend.USEARCH.
        :param **kwargs: Any additional keyword arguments to pass to the Vicinity index.
        :return: A SemHash instance with a fitted vicinity index.
        :raises ValueError: If records are empty.
        :raises ValueError: If the number of embeddings doesn't match the number of records.
        :raises ValueError: If embeddings is not a 2D array.
        :raises ValueError: If columns are not provided for dictionary records.
        """
        if len(records) == 0:
            raise ValueError("records must not be empty")

        embeddings = np.asarray(embeddings)
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be a 2D array, got shape {embeddings.shape}")

        if embeddings.shape[0] != len(records):
            raise ValueError(
                f"Number of embeddings ({embeddings.shape[0]}) must match number of records ({len(records)})"
            )

        # Prepare and validate records
        dict_records, columns, was_string = prepare_records(records, columns)

        column_set = set(columns)

        # Group exact duplicates (by columns) and keep only the first embedding per group
        items: list[list[dict[str, str]]] = []
        keep_embedding_indices: list[int] = []
        key_to_item_idx: dict[frozendict[str, str], int] = {}

        for i, record in enumerate(dict_records):
            key = to_frozendict(record, column_set)
            item_idx = key_to_item_idx.get(key)
            if item_idx is None:
                key_to_item_idx[key] = len(items)
                items.append([record])
                keep_embedding_indices.append(i)
            else:
                items[item_idx].append(record)

        deduplicated_embeddings = embeddings[keep_embedding_indices]

        index = Index.from_vectors_and_items(
            vectors=deduplicated_embeddings, items=items, backend_type=ann_backend, **kwargs
        )
        return cls(index=index, model=model, columns=columns, was_string=was_string)

    def deduplicate(
        self,
        records: Sequence[Record],
        threshold: float = 0.9,
    ) -> DeduplicationResult:
        """
        Perform deduplication against the fitted index.

        This method assumes you have already fit on a reference dataset (e.g., a train set) with from_records.
        It will remove any items from 'records' that are similar above a certain threshold
        to any item in the fitted dataset.

        :param records: A new set of records (e.g., test set) to deduplicate against the fitted dataset.
        :param threshold: Similarity threshold for deduplication.
        :return: A deduplicated list of records.
        """
        dict_records = self._validate_if_strings(records)

        # Remove exact duplicates before embedding
        dict_records, exact_duplicates = remove_exact_duplicates(
            records=dict_records, columns=self.columns, reference_records=self.index.items
        )
        duplicate_records = []
        for record, duplicates in exact_duplicates:
            duplicated_with_score = add_scores_to_records(duplicates)
            duplicate_record = DuplicateRecord(record=record, duplicates=duplicated_with_score, exact=True)
            duplicate_records.append(duplicate_record)

        # If no records are left after removing exact duplicates, return early
        if not dict_records:
            return DeduplicationResult(
                selected=[], filtered=duplicate_records, threshold=threshold, columns=self.columns
            )

        # Compute embeddings for the new records
        embeddings = featurize(records=dict_records, columns=self.columns, model=self.model)
        # Query the fitted index
        results = self.index.query_threshold(embeddings, threshold=threshold)

        deduplicated_records = []
        for record, similar_items in zip(dict_records, results):
            if not similar_items:
                # No duplicates found, keep this record
                deduplicated_records.append(record)
            else:
                duplicate_records.append(
                    DuplicateRecord(
                        record=record,
                        duplicates=[(item, score) for item, score in similar_items],
                        exact=False,
                    )
                )

        result = DeduplicationResult(
            selected=deduplicated_records, filtered=duplicate_records, threshold=threshold, columns=self.columns
        )

        if self._was_string:
            # Convert records back to strings if the records were originally strings
            return map_deduplication_result_to_strings(result, columns=self.columns)

        return result

    def self_deduplicate(
        self,
        threshold: float = 0.9,
    ) -> DeduplicationResult:
        """
        Deduplicate within the same dataset. This can be used to remove duplicates from a single dataset.

        :param threshold: Similarity threshold for deduplication.
        :return: A deduplicated list of records.
        """
        # Query the fitted index
        results = self.index.query_threshold(self.index.vectors, threshold=threshold)
        column_set = set(self.columns)

        duplicate_records = []

        deduplicated_records = []
        seen_items: set[frozendict[str, str]] = set()
        for item, similar_items in zip(self.index.items, results):
            # Items is a list of items which are exact duplicates of each other
            # So if the an item has more than one record, it is an exact duplicate
            # Crucially, we should count each instance separately.
            record, *duplicates = item
            # We need to compare all duplicates to all _items_.
            # The first item in a list of duplicate is not duplicated, because otherwise
            # we would remove the whole cluster. But it is a duplicate for the other items.

            # Iterate from index 1.
            for index, curr_record in enumerate(duplicates, 1):
                # The use of indexing is intentional here, we want to check if the object is the same
                # not if they have the same values. If we did != or is we would probably ignore lots
                # of items.
                items_to_keep = item[:index] + item[index + 1 :]
                items_with_score = add_scores_to_records(items_to_keep)
                duplicate_records.append(DuplicateRecord(record=curr_record, duplicates=items_with_score, exact=True))

            # If we don't see any similar_items, we know the record is not a duplicate.
            # In rare cases, the item itself might not be returned by the index.
            if not similar_items:  # pragma: no cover
                deduplicated_records.append(record)
                continue
            items, _ = zip(*similar_items)
            frozen_items = [to_frozendict(item, column_set) for item in items]
            # similar_items includes 'record' itself
            # If we've seen any of these items before, this is a duplicate cluster.
            if any(item in seen_items for item in frozen_items):
                duplicate_records.append(
                    DuplicateRecord(
                        record=record,
                        duplicates=[(item, score) for item, score in similar_items if item != record],
                        exact=False,
                    )
                )
                continue
            # This is the first time we see this cluster of similar items
            deduplicated_records.append(record)
            # Mark all items in this cluster as seen
            seen_items.update(frozen_items)

        result = DeduplicationResult(
            selected=deduplicated_records, filtered=duplicate_records, threshold=threshold, columns=self.columns
        )

        if self._was_string:
            # Convert records back to strings if the records were originally strings
            return map_deduplication_result_to_strings(result, columns=self.columns)

        return result

    def _validate_if_strings(self, records: Sequence[dict[str, Any] | str]) -> list[dict[str, Any]]:
        """
        Validate if the records are strings.

        If the records are strings, they are converted to dictionaries with a single column.
        If the records are dicts, primitives are stringified and complex types (images, etc.) are kept raw.

        :param records: The records to validate.
        :return: The records as a list of dictionaries.
        :raises ValueError: If records are empty.
        :raises ValueError: If the records are strings but were not originally strings.
        :raises ValueError: If the records are not all strings or all dictionaries.
        :raises ValueError: If dict record contains None values.
        """
        if len(records) == 0:
            raise ValueError("records must not be empty")

        # String path
        if isinstance(records[0], str):
            if not self._was_string:
                raise ValueError("Records were not originally strings, but you passed strings.")
            if not all(isinstance(r, str) for r in records):
                raise ValueError("Records must be all strings.")
            return [{"text": r} for r in records]

        # Dict path
        if not all(isinstance(r, dict) for r in records):
            raise ValueError("Records must be all dictionaries.")

        dict_records: Sequence[dict[str, Any]] = records  # type: ignore[assignment]
        result: list[dict[str, Any]] = []
        for record in dict_records:
            # Start with a copy of the full record to preserve non-embedding fields
            out = dict(record)
            # Then coerce only the embedding columns
            for col in self.columns:
                if (val := record.get(col)) is None:
                    raise ValueError(f"Column '{col}' has None value in record {record}")
                out[col] = coerce_value(val)
            result.append(out)
        return result

    def find_representative(
        self,
        records: Sequence[Record],
        selection_size: int = 10,
        candidate_limit: int | Literal["auto"] = "auto",
        diversity: float = 0.5,
        strategy: Strategy | str = Strategy.MMR,
    ) -> FilterResult:
        """
        Find representative samples from a given set of records against the fitted index.

        First, the records are ranked using average similarity.
        Then, the top candidates are re-ranked using Maximal Marginal Relevance (MMR)
        to select a diverse set of representatives.

        :param records: The records to rank and select representatives from.
        :param selection_size: Number of representatives to select.
        :param candidate_limit: Number of top candidates to consider for diversity reranking.
            Defaults to "auto", which calculates the limit based on the total number of records.
        :param diversity: Trade-off between diversity (1.0) and relevance (0.0). Default is 0.5.
        :param strategy: Diversification strategy (MMR, MSD, DPP, COVER, SSD). Default is MMR.
        :return: A FilterResult with the diversified candidates.
        """
        ranking = self._rank_by_average_similarity(records)
        if candidate_limit == "auto":
            candidate_limit = compute_candidate_limit(total=len(ranking.selected), selection_size=selection_size)
        return self._diversify(ranking, candidate_limit, selection_size, diversity, strategy)

    def self_find_representative(
        self,
        selection_size: int = 10,
        candidate_limit: int | Literal["auto"] = "auto",
        diversity: float = 0.5,
        strategy: Strategy | str = Strategy.MMR,
    ) -> FilterResult:
        """
        Find representative samples from the fitted dataset.

        First, the rank the records are ranked using average similarity.
        Then, the top candidates are re-ranked using Maximal Marginal Relevance (MMR)
        to select a diverse set of representatives.

        :param selection_size: Number of representatives to select.
        :param candidate_limit: Number of top candidates to consider for diversity reranking.
            Defaults to "auto", which calculates the limit based on the total number of records.
        :param diversity: Trade-off between diversity (1.0) and relevance (0.0). Default is 0.5.
        :param strategy: Diversification strategy (MMR, MSD, DPP, COVER, SSD). Default is MMR.
        :return: A FilterResult with the diversified representatives.
        """
        ranking = self._self_rank_by_average_similarity()
        if candidate_limit == "auto":
            candidate_limit = compute_candidate_limit(total=len(ranking.selected), selection_size=selection_size)
        return self._diversify(ranking, candidate_limit, selection_size, diversity, strategy)

    def filter_outliers(
        self,
        records: Sequence[Record],
        outlier_percentage: float = 0.1,
    ) -> FilterResult:
        """
        Filter outliers in a given set of records against the fitted dataset.

        This method ranks the records by their average similarity and filters the bottom
        outlier_percentage of records as outliers.

        :param records: A sequence of records to find outliers in.
        :param outlier_percentage: The percentage (between 0 and 1) of records to consider outliers.
        :return: A FilterResult where 'selected' contains the inliers and 'filtered' contains the outliers.
        :raises ValueError: If outlier_percentage is not between 0 and 1.
        """
        if outlier_percentage < 0.0 or outlier_percentage > 1.0:
            raise ValueError("outlier_percentage must be between 0 and 1")
        ranking = self._rank_by_average_similarity(records)
        outlier_count = ceil(len(ranking.selected) * outlier_percentage)
        if outlier_count == 0:
            # If the outlier count is 0, return no outliers
            return FilterResult(
                selected=ranking.selected,
                filtered=[],
                scores_selected=ranking.scores_selected,
                scores_filtered=[],
            )

        outlier_records = ranking.selected[-outlier_count:]
        outlier_scores = ranking.scores_selected[-outlier_count:]
        inlier_records = ranking.selected[:-outlier_count]
        inlier_scores = ranking.scores_selected[:-outlier_count]

        return FilterResult(
            selected=inlier_records,
            filtered=outlier_records,
            scores_selected=inlier_scores,
            scores_filtered=outlier_scores,
        )

    def self_filter_outliers(
        self,
        outlier_percentage: float = 0.1,
    ) -> FilterResult:
        """
        Filter outliers in the fitted dataset.

        The method ranks the records stored in the index and filters the bottom outlier_percentage
        of records as outliers.

        :param outlier_percentage: The percentage (between 0 and 1) of records to consider as outliers.
        :return: A FilterResult where 'selected' contains the inliers and 'filtered' contains the outliers.
        :raises ValueError: If outlier_percentage is not between 0 and 1.
        """
        if outlier_percentage < 0.0 or outlier_percentage > 1.0:
            raise ValueError("outlier_percentage must be between 0 and 1")
        ranking = self._self_rank_by_average_similarity()
        outlier_count = ceil(len(ranking.selected) * outlier_percentage)
        if outlier_count == 0:
            # If the outlier count is 0, return no outliers
            return FilterResult(
                selected=ranking.selected,
                filtered=[],
                scores_selected=ranking.scores_selected,
                scores_filtered=[],
            )

        outlier_records = ranking.selected[-outlier_count:]
        outlier_scores = ranking.scores_selected[-outlier_count:]
        inlier_records = ranking.selected[:-outlier_count]
        inlier_scores = ranking.scores_selected[:-outlier_count]

        return FilterResult(
            selected=inlier_records,
            filtered=outlier_records,
            scores_selected=inlier_scores,
            scores_filtered=outlier_scores,
        )

    def _rank_by_average_similarity(
        self,
        records: Sequence[Record],
    ) -> FilterResult:
        """
        Rank a given set of records based on the average cosine similarity of the neighbors in the fitted index.

        :param records: A sequence of records.
        :return: A FilterResult containing the ranking (records sorted and their average similarity scores).
        """
        dict_records = self._validate_if_strings(records)
        embeddings = featurize(records=dict_records, columns=self.columns, model=self.model)
        results = self.index.query_top_k(embeddings, k=100, vectors_are_in_index=False)

        # Compute the average similarity for each record.
        sorted_scores = sorted(
            ((record, np.mean(sims)) for record, (_, sims) in zip(dict_records, results)),
            key=lambda x: x[1],
            reverse=True,
        )
        selected, scores_selected = zip(*sorted_scores)

        return FilterResult(
            selected=list(selected),
            filtered=[],
            scores_selected=list(scores_selected),
            scores_filtered=[],
        )

    def _self_rank_by_average_similarity(
        self,
    ) -> FilterResult:
        """
        Rank the records stored in the fitted index based on the average cosine similarity of the neighbors.

        :return: A FilterResult containing the ranking.
        """
        if self._ranking_cache is not None:
            return self._ranking_cache

        dict_records = [record[0] for record in self.index.items]
        results = self.index.query_top_k(self.index.vectors, k=100, vectors_are_in_index=True)

        # Compute the average similarity for each record.
        sorted_scores = sorted(
            ((record, np.mean(sims)) for record, (_, sims) in zip(dict_records, results)),
            key=lambda x: x[1],
            reverse=True,
        )
        selected, scores_selected = zip(*sorted_scores)

        ranking = FilterResult(
            selected=list(selected),
            filtered=[],
            scores_selected=list(scores_selected),
            scores_filtered=[],
        )
        self._ranking_cache = ranking
        return ranking

    def _diversify(
        self,
        ranked_results: FilterResult,
        candidate_limit: int,
        selection_size: int,
        diversity: float,
        strategy: Strategy | str = Strategy.MMR,
    ) -> FilterResult:
        """Diversify top candidates using the specified strategy."""
        candidates = ranked_results.selected[:candidate_limit]
        relevance = ranked_results.scores_selected[:candidate_limit]

        if not candidates:
            return FilterResult(selected=[], filtered=[], scores_selected=[], scores_filtered=[])

        embeddings = featurize(records=candidates, columns=self.columns, model=self.model)
        result = diversify(
            embeddings=embeddings,
            scores=np.array(relevance),
            k=selection_size,
            strategy=strategy,
            diversity=diversity,
        )

        selected_set = set(result.indices)
        return FilterResult(
            selected=[candidates[i] for i in result.indices],
            filtered=[rec for i, rec in enumerate(candidates) if i not in selected_set],
            scores_selected=result.selection_scores.tolist(),
            scores_filtered=[score for i, score in enumerate(relevance) if i not in selected_set],
        )
