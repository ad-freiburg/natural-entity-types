"""
Copyright 2019, University of Freiburg
Chair of Algorithms and Data Structures.
Claudius Korzen <korzen@cs.uni-freiburg.de>
Patrick Brosi <brosi@cs.uni-freiburg.de>
Natalie Prange <prange@cs.uni-freiburg.de>
"""


class Metrics:
    @staticmethod
    def precision_at_k(result_ids, relevant_ids, k):
        """
        Compute the measure P@k for the given list of result ids as it was
        returned by the inverted index for a single query, and the given set of
        relevant document ids.

        Note that the relevant document ids are 1-based (as they reflect the
        line number in the dataset file).

        >>> metrics = Metrics()
        >>> metrics.precision_at_k([5, 3, 6, 1, 2], {1, 2, 5, 6, 7, 8}, k=0)
        0
        >>> metrics.precision_at_k([5, 3, 6, 1, 2], {1, 2, 5, 6, 7, 8}, k=4)
        0.75
        >>> metrics.precision_at_k([5, 3, 6, 1, 2], {1, 2, 5, 6, 7, 8}, k=8)
        0.5
        """
        if k == 0:
            return 0

        num_relevant_result_ids = 0
        for i in range(0, min(len(result_ids), k)):
            if result_ids[i] in relevant_ids:
                num_relevant_result_ids += 1
        return num_relevant_result_ids / k

    @staticmethod
    def average_precision(result_ids, relevant_ids):
        """
        Compute the average precision (AP) for the given list of result ids as
        it was returned by the inverted index for a single query, and the given
        set of relevant document ids.

        Note that the relevant document ids are 1-based (as they reflect the
        line number in the dataset file).

        >>> metrics = Metrics()
        >>> metrics.average_precision([7, 17, 9, 42, 5], {5, 7, 12, 42})
        0.525
        """
        sum_ap = 0
        for i in range(0, len(result_ids)):
            if result_ids[i] in relevant_ids:
                sum_ap += Metrics.precision_at_k(result_ids, relevant_ids, i + 1)
        return sum_ap / len(relevant_ids)