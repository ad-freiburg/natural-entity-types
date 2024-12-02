from enum import Enum


class MetricName(Enum):
    HIT_RATE_AT_1 = "Hit Rate @ 1"
    HIT_RATE_AT_3 = "Hit Rate @ 3"
    HIT_RATE_AT_5 = "Hit Rate @ 5"
    HIT_RATE_AT_10 = "Hit Rate @ 10"
    MRR = "MRR"
    AVERAGE_PRECISION = "MAP"
    PRECISION_AT_R = "MP@R"


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

    @staticmethod
    def hit_rate_at_k(result_ids, relevant_ids, k):
        for i in range(min(k, len(result_ids))):
            if result_ids[i] in relevant_ids:
                return 1
        return 0

    @staticmethod
    def mrr(result_ids, relevant_ids):
        # Compute the mean reciprocal rank (MRR) for the given list of result ids
        for i in range(len(result_ids)):
            if result_ids[i] in relevant_ids:
                return 1 / (i + 1)
        return 0