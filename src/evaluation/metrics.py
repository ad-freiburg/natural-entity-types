from enum import Enum


class MetricName(Enum):
    TOP_1_ACCURACY = "Top-1 Accuracy"
    TOP_3_ACCURACY = "Top-3 Accuracy"
    TOP_5_ACCURACY = "Top-5 Accuracy"
    TOP_10_ACCURACY = "Top-10 Accuracy"
    MRR = "MRR"
    AVERAGE_PRECISION = "MAP"
    PRECISION_AT_R = "MP@R"


class Metrics:
    @staticmethod
    def precision_at_k(result_ids, relevant_ids, k):
        if k == 0:
            return 0

        num_relevant_result_ids = 0
        for i in range(0, min(len(result_ids), k)):
            if result_ids[i] in relevant_ids:
                num_relevant_result_ids += 1
        return num_relevant_result_ids / k

    @staticmethod
    def average_precision(result_ids, relevant_ids):
        sum_ap = 0
        for i in range(0, len(result_ids)):
            if result_ids[i] in relevant_ids:
                sum_ap += Metrics.precision_at_k(result_ids, relevant_ids, i + 1)
        return sum_ap / len(relevant_ids)

    @staticmethod
    def top_k_accuracy(result_ids, relevant_ids, k):
        for i in range(min(k, len(result_ids))):
            if result_ids[i] in relevant_ids:
                return 1
        return 0

    @staticmethod
    def mrr(result_ids, relevant_ids):
        """
        Compute the mean reciprocal rank (MRR) for the given list of result ids.
        """
        for i in range(len(result_ids)):
            if result_ids[i] in relevant_ids:
                return 1 / (i + 1)
        return 0