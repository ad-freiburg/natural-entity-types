from src.evaluation.metrics import Metrics, MetricName
from src.utils.colors import Colors


def get_result_for_metric(metric, result_types, benchmark_types):
    if metric == MetricName.HIT_RATE_AT_1:
        evaluation_result = Metrics.hit_rate_at_k(result_types, benchmark_types, 1)
    elif metric == MetricName.HIT_RATE_AT_3:
        evaluation_result = Metrics.hit_rate_at_k(result_types, benchmark_types, 3)
    elif metric == MetricName.HIT_RATE_AT_5:
        evaluation_result = Metrics.hit_rate_at_k(result_types, benchmark_types, 5)
    elif metric == MetricName.HIT_RATE_AT_10:
        evaluation_result = Metrics.hit_rate_at_k(result_types, benchmark_types, 10)
    elif metric == MetricName.MRR:
        evaluation_result = Metrics.mrr(result_types, benchmark_types)
    elif metric == MetricName.AVERAGE_PRECISION:
        evaluation_result = Metrics.average_precision(result_types, benchmark_types)
    elif metric == MetricName.PRECISION_AT_R:
        evaluation_result = Metrics.precision_at_k(result_types, benchmark_types, len(benchmark_types))
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return evaluation_result


def evaluate(scoring_function, benchmark, metrics, entity_db=None, verbose=False):
    evaluation_results = {metric: [] for metric in metrics}
    for entity_id in benchmark:
        result_types = scoring_function(entity_id)
        if result_types is None or len(result_types) == 0 or result_types[0] is None:
            result_types = ([None])
        elif type(result_types[0]) is tuple:
            result_types = [r[1] for r in result_types]  # Get only the type ids, not the scores

        for metric in metrics:
            evaluation_result = get_result_for_metric(metric, result_types, benchmark[entity_id])
            evaluation_results[metric].append(evaluation_result)

        if verbose:
            entity_name = entity_db.get_entity_name(entity_id)
            gt_entities = ", ".join([f"{Colors.BLUE}{entity_db.get_entity_name(t)}{Colors.END} ({t})" for t in benchmark[entity_id]])
            predicted_entities = ", ".join([f"{Colors.BLUE}{entity_db.get_entity_name(t)}{Colors.END} ({t})" for t in result_types[:4]]) + "..."
            results = ", ".join([f"{metric.value}: {evaluation_results[metric][-1]}" for metric in evaluation_results])
            print()
            print(f"Results for {Colors.BOLD}\"{entity_name}\"{Colors.END} ({entity_id}):\n"
                  f"\tGround truth: {gt_entities}\n"
                  f"\tPrediction: {predicted_entities}\n"
                  f"\tResults: {results}")

    mean_evaluation_results = {}
    for metric in evaluation_results:
        mean_evaluation_results[metric] = sum(evaluation_results[metric]) / len(evaluation_results[metric])

    return mean_evaluation_results


def evaluate_batch_prediction(y_pred, benchmark, entity_index, metrics, entity_db=None, verbose=False):
    evaluation_results = {metric: [] for metric in metrics}
    for entity_id in benchmark:
        if entity_id in entity_index:
            indices = entity_index[entity_id]
            result_types = [(y_pred[i], t) for i, t in indices]
            result_types = sorted(result_types, key=lambda x: x[0], reverse=True)
        else:
            result_types = []
        if result_types is None or len(result_types) == 0 or result_types[0] is None:
            result_types = ([None])
        elif type(result_types[0]) is tuple:
            result_types = [r[1] for r in result_types]  # Get only the type ids, not the scores

        for metric in metrics:
            evaluation_result = get_result_for_metric(metric, result_types, benchmark[entity_id])
            evaluation_results[metric].append(evaluation_result)

        if verbose:
            entity_name = entity_db.get_entity_name(entity_id)
            gt_entities = ", ".join([f"{entity_db.get_entity_name(t)} ({t})" for t in benchmark[entity_id]])
            predicted_entities = ", ".join([f"{entity_db.get_entity_name(t)} ({t})" for t in result_types[:6]]) + "..."
            results = ", ".join([f"{metric}: {evaluation_results[metric]}" for metric in evaluation_results])
            print(f"Results for \"{entity_name}\" ({entity_id}):\n"
                  f"\tGround truth: {gt_entities}\n"
                  f"\tPrediction: {predicted_entities}"
                  f"\tResults: {results}")

    mean_evaluation_results = {}
    for metric in evaluation_results:
        mean_evaluation_results[metric] = sum(evaluation_results[metric]) / len(evaluation_results[metric])

    return mean_evaluation_results