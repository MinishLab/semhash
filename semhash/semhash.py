from __future__ import annotations

from collections import defaultdict
from math import ceil
from typing import Generic, Literal, Sequence

import numpy as np
from frozendict import frozendict
from model2vec import StaticModel
from vicinity import Backend

from semhash.datamodels import DeduplicationResult, DuplicateRecord, FilterResult, Record
from semhash.index import Index
from semhash.records import add_scores_to_records, map_deduplication_result_to_strings
from semhash.utils import Encoder, compute_candidate_limit, to_frozendict


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

    @staticmethod
    def _featurize(
        records: Sequence[dict[str, str]],
        columns: Sequence[str],
        model: Encoder,
    ) -> np.ndarray:
        """
        Featurize a list of records using the model.

        :param records: A list of records.
        :param columns: Columns to featurize.
        :param model: An Encoder model.
        :return: The embeddings of the records.
        """
        # Extract the embeddings for each column across all records
        embeddings_per_col = []
        for col in columns:
            col_texts = [r[col] for r in records]
            col_emb = model.encode(col_texts)
            embeddings_per_col.append(np.asarray(col_emb))

        return np.concatenate(embeddings_per_col, axis=1)

    @classmethod
    def _remove_exact_duplicates(
        cls,
        records: Sequence[dict[str, str]],
        columns: Sequence[str],
        reference_records: list[list[dict[str, str]]] | None = None,
    ) -> tuple[list[dict[str, str]], list[tuple[dict[str, str], list[dict[str, str]]]]]:
        """
        Remove exact duplicates based on the unpacked string representation of each record.

        If reference_records is None, the function will only check for duplicates within the records list.

        :param records: A list of records to check for exact duplicates.
        :param columns: Columns to unpack.
        :param reference_records: A list of records to compare against. These are already unpacked
        :return: A list of deduplicated records and a list of duplicates.
        """
        deduplicated = []
        duplicates = []

        column_set = set(columns)
        # Build a seen set from reference_records if provided
        seen: defaultdict[frozendict[str, str], list[dict[str, str]]] = defaultdict(list)
        if reference_records is not None:
            for record_set in reference_records:
                key = to_frozendict(record_set[0], column_set)
                seen[key] = list(record_set)
        in_one_set = reference_records is None

        for record in records:
            frozen_record = frozendict({k: v for k, v in record.items() if k in column_set})
            if duplicated_records := seen.get(frozen_record):
                duplicates.append((record, duplicated_records))
            else:
                deduplicated.append(record)
                # Only add current documents to seen if no reference set is used
                if in_one_set:
                    seen[frozen_record].append(record)

        return deduplicated, duplicates

    @classmethod
    def from_records(
        cls,
        records: Sequence[Record],
        columns: Sequence[str] | None = None,
        use_ann: bool = True,
        model: Encoder | None = None,
    ) -> SemHash:
        """
        Initialize a SemHash instance from records.

        This removes exact duplicates, featurizes the records, and fits a vicinity index.

        :param records: A list of records (strings or dictionaries).
        :param columns: Columns to featurize if records are dictionaries.
        :param use_ann: Whether to use approximate nearest neighbors (True) or basic search (False). Default is True.
        :param model: (Optional) An Encoder model. If None, the default model is used (minishlab/potion-base-8M).
        :return: A SemHash instance with a fitted vicinity index.
        :raises ValueError: If columns are not provided for dictionary records.
        """
        if columns is None and isinstance(records[0], dict):
            raise ValueError("Columns must be specified when passing dictionaries.")

        if isinstance(records[0], str):
            # If records are strings, convert to dictionaries with a single column
            columns = ["text"]
            dict_records: list[dict[str, str]] = [{"text": record} for record in records]
            was_string = True
        else:
            dict_records = list(records)
            was_string = False

        # If no model is provided, load the default model
        if model is None:
            model = StaticModel.from_pretrained("minishlab/potion-base-8M")

        # Remove exact duplicates
        deduplicated_records, duplicates = cls._remove_exact_duplicates(dict_records, columns)

        col_set = set(columns)
        duplicate_map = defaultdict(list)
        for x, _ in duplicates:
            frozen_record = to_frozendict(x, col_set)
            duplicate_map[frozen_record].append(x)

        items: list[list[dict[str, str]]] = []
        for record in deduplicated_records:
            i = [record]
            frozen_record = to_frozendict(record, set(columns))
            i.extend(duplicate_map[frozen_record])
            items.append(i)

        # Create embeddings and unpack records
        embeddings = cls._featurize(deduplicated_records, columns, model)

        # Build the Vicinity index
        backend = Backend.USEARCH if use_ann else Backend.BASIC
        index = Index.from_vectors_and_items(
            vectors=embeddings,
            items=items,
            backend_type=backend,
        )

        return cls(index=index, columns=columns, model=model, was_string=was_string)

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
        dict_records, exact_duplicates = self._remove_exact_duplicates(
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
                deduplicated=[], duplicates=duplicate_records, threshold=threshold, columns=self.columns
            )

        # Compute embeddings for the new records
        embeddings = self._featurize(records=dict_records, columns=self.columns, model=self.model)
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
            deduplicated=deduplicated_records, duplicates=duplicate_records, threshold=threshold, columns=self.columns
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
            # in rare cases, the item itself might not be a duplicate of itself.
            if not similar_items:
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
            deduplicated=deduplicated_records, duplicates=duplicate_records, threshold=threshold, columns=self.columns
        )

        if self._was_string:
            # Convert records back to strings if the records were originally strings
            return map_deduplication_result_to_strings(result, columns=self.columns)

        return result

    def _validate_if_strings(self, records: Sequence[dict[str, str] | str]) -> Sequence[dict[str, str]]:
        """
        Validate if the records are strings.

        If the records are strings, they are converted to dictionaries with a single column.

        :param records: The records to validate.
        :return: The records as a list of dictionaries.
        :raises ValueError: If the records are strings but were not originally strings.
        :raises ValueError: If the records are not all strings or dictionaries.
        """
        if isinstance(records[0], str):
            if not self._was_string:
                raise ValueError("Records were not originally strings, but you passed strings.")
            dict_records = [{"text": record} for record in records if isinstance(record, str)]
        else:
            dict_records = [record for record in records if isinstance(record, dict)]
        if len(dict_records) != len(records):
            raise ValueError("Records must be either strings or dictionaries.")
        return dict_records

    def find_representative(
        self,
        records: Sequence[Record],
        selection_size: int = 10,
        candidate_limit: int | Literal["auto"] = "auto",
        lambda_param: float = 0.5,
    ) -> FilterResult:
        """
        Find representative samples from a given set of records against the fitted index.

        First, the records are ranked using average similarity.
        Then, the top candidates are re-ranked using Maximal Marginal Relevance (MMR)
        to select a diverse set of representatives.

        :param records: The records to rank and select representatives from.
        :param selection_size: Number of representatives to select.
        :param candidate_limit: Number of top candidates to consider for MMR reranking.
            Defaults to "auto", which calculates the limit based on the total number of records.
        :param lambda_param: Trade-off parameter between relevance (1.0) and diversity (0.0). Must be between 0 and 1.
        :return: A FilterResult with the diversified candidates.
        """
        ranking = self._rank_by_average_similarity(records)
        if candidate_limit == "auto":
            candidate_limit = compute_candidate_limit(total=len(ranking.selected), selection_size=selection_size)
        return self._mmr(ranking, candidate_limit, selection_size, lambda_param)

    def self_find_representative(
        self,
        selection_size: int = 10,
        candidate_limit: int | Literal["auto"] = "auto",
        lambda_param: float = 0.5,
    ) -> FilterResult:
        """
        Find representative samples from the fitted dataset.

        First, the rank the records are ranked using average similarity.
        Then, the top candidates are re-ranked using Maximal Marginal Relevance (MMR)
        to select a diverse set of representatives.

        :param selection_size: Number of representatives to select.
        :param candidate_limit: Number of top candidates to consider for MMR reranking.
            Defaults to "auto", which calculates the limit based on the total number of records.
        :param lambda_param: Trade-off parameter between relevance (1.0) and diversity (0.0). Must be between 0 and 1.
        :return: A FilterResult with the diversified representatives.
        """
        ranking = self._self_rank_by_average_similarity()
        if candidate_limit == "auto":
            candidate_limit = compute_candidate_limit(total=len(ranking.selected), selection_size=selection_size)
        return self._mmr(ranking, candidate_limit, selection_size, lambda_param)

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
            # If the outlier count is 0, return an empty selection
            return FilterResult(
                selected=[], filtered=ranking.selected, scores_selected=[], scores_filtered=ranking.scores_selected
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
            # If the outlier count is 0, return an empty selection
            return FilterResult(
                selected=[], filtered=ranking.selected, scores_selected=[], scores_filtered=ranking.scores_selected
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
        embeddings = self._featurize(records=dict_records, columns=self.columns, model=self.model)
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

    def _mmr(
        self,
        ranked_results: FilterResult,
        candidate_limit: int,
        selection_size: int,
        lambda_param: float,
    ) -> FilterResult:
        """
        Perform Maximal Marginal Relevance (MMR) re-ranking on the top candidates from a FilterResult.

        This function first slices the ranking, then computes embeddings for the candidates,
        normalizes them, and finally performs MMR re-ranking to obtain a diverse subset of representatives.

        :param ranked_results: A FilterResult containing the ranking of records.
        :param candidate_limit: Number of top candidates to consider from the ranking.
        :param selection_size: Number of final candidates to select using MMR.
        :param lambda_param: Weight balancing relevance and diversity (between 0 and 1).
        :return: A FilterResult containing the selected (diversified) representatives along with
                their MMR scores in `scores_selected`. The remaining candidates are placed in filtered.
        :raises ValueError: If lambda_param is not between 0 and 1.
        """
        if not (0.0 <= lambda_param <= 1.0):
            raise ValueError("lambda_param must be between 0 and 1")

        # Slice the top candidates from the ranking.
        candidate_records = ranked_results.selected[:candidate_limit]
        candidate_relevance = ranked_results.scores_selected[:candidate_limit]

        # Compute embeddings for candidate records.
        embeddings = self._featurize(records=candidate_records, columns=self.columns, model=self.model)

        # Normalize embeddings for cosine similarity.
        normalized_embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-16)

        # Package candidates as tuples: (record, baseline_relevance, normalized_embedding)
        candidates = [
            (record, relevance, normalized_embedding)
            for record, relevance, normalized_embedding in zip(
                candidate_records, candidate_relevance, normalized_embeddings
            )
        ]

        # If no candidates, return empty result.
        if not candidates:
            return FilterResult(selected=[], filtered=[], scores_selected=[], scores_filtered=[])

        # Initialize selected set with the most relevant candidate.
        first_record, first_relevance, first_embedding = candidates[0]
        selected_records = [first_record]
        selected_scores = [first_relevance]
        selected_embeddings = [first_embedding]
        remaining_candidates = candidates[1:]

        # Iteratively select candidates using the MMR criterion.
        while remaining_candidates and len(selected_records) < selection_size:
            # Build arrays for the remaining candidates.
            embeddings_remaining = np.vstack([emb for (_, _, emb) in remaining_candidates])
            relevances_remaining = np.array([rel for (_, rel, _) in remaining_candidates])

            # Build array of embeddings for the already selected set.
            embeddings_selected = np.vstack(selected_embeddings)

            # Compute cosine similarities between selected and remaining candidates.
            similarity_matrix = embeddings_remaining.dot(embeddings_selected.T)
            max_similarity = similarity_matrix.max(axis=1)

            # Compute MMR scores for all remaining candidates.
            mmr_scores = lambda_param * relevances_remaining - (1.0 - lambda_param) * max_similarity

            # Choose the candidate with the highest MMR score.
            best_index = int(np.argmax(mmr_scores))
            record, _, embedding = remaining_candidates.pop(best_index)

            selected_records.append(record)
            selected_scores.append(mmr_scores[best_index])
            selected_embeddings.append(embedding)

        # Whatever is left in remaining_candidates is filtered out.
        filtered_records = [rec for (rec, _, _) in remaining_candidates]
        filtered_scores = [rel for (_, rel, _) in remaining_candidates]

        return FilterResult(
            selected=selected_records,
            filtered=filtered_records,
            scores_selected=selected_scores,
            scores_filtered=filtered_scores,
        )
