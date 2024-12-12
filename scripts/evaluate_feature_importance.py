"""
Evaluate the feature importance of the neural network model by shuffling the values of each feature and reporting
the model's performance.
"""
import argparse
import sys

sys.path.append(".")

from src.utils import log
from src.type_computation.neural_network import NeuralTypePredictor
from src.evaluation.benchmark_reader import BenchmarkReader
from src.evaluation.evaluation import evaluate_batch_prediction
from src.evaluation.metrics import MetricName
from src.utils.colors import Colors


def main(args):
    logger.info("Initializing Neural Network ...")
    nn = NeuralTypePredictor()
    nn.load_model(args.load_model)

    column_ranges = [None, (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 308), (308, 608),
                     (0, 3), (0, 8), (8, 608)]
    feature_names = ["None", "norm_pop", "norm_var", "norm_idf", "path_length", "type_in_desc", "len_type_name",
                     "len_desc", "type_in_label", "desc_emb", "type_label_emb", "computed features",
                     "all_but_embeddings", "embeddings"]
    for i, column_range in enumerate(column_ranges):
        logger.info(f"Evaluating feature importance for feature {Colors.BOLD}{feature_names[i]}{Colors.END}")
        for benchmark_file in args.benchmark_files:
            benchmark = BenchmarkReader.read_benchmark(benchmark_file)
            X, y, entity_index = nn.create_dataset(benchmark_file, column_range, True)
            logger.info(f"Evaluating feature importance on benchmark {Colors.BLUE}{benchmark_file}{Colors.END}")
            y_pred = nn.predict_batch(X)
            evaluation_results = evaluate_batch_prediction(y_pred, benchmark, entity_index,
                                                           [MetricName.TOP_1_ACCURACY])
            for metric in evaluation_results:
                print(f"{metric.value}: {evaluation_results[metric]*100:.1f}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    parser.add_argument("--load_model", type=str, help="File from which to load the model.")
    parser.add_argument("-b", "--benchmark_files", type=str, required=True, nargs='+',
                        help="File that contains the benchmark.")

    logger = log.setup_logger()

    main(parser.parse_args())
