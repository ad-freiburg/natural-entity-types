import argparse
import logging
import sys
from itertools import product

sys.path.append(".")

from src.models.entity_database import EntityDatabase
from src.type_computation.neural_network import NeuralTypePredictor
from src.evaluation.benchmark_reader import BenchmarkReader
from src.evaluation.metrics import MetricName
from src.utils import log
from src.evaluation.evaluation import evaluate
from src.utils.colors import Colors


def print_parameters(params, parameter_names, top_k=1):
    print()
    for name in parameter_names:
        print(f"{name}: ", end="")
        for i in range(min(top_k, len(params))):
            print(f"\t{params[i][name]}", end="")
        print()
    print()


def main(args):
    entity_db = EntityDatabase()
    entity_db.load_instance_of_mapping()
    entity_db.load_subclass_of_mapping()
    entity_db.load_entity_to_name()
    entity_db.load_entity_to_description()
    nn = NeuralTypePredictor(entity_db)

    parameters = {
        "optimizer": ["adam"],
        "hidden_layer_size": [256, 512],
        "activation": ["tanh", "sigmoid", "relu", "leaky_relu"],
        "dropout": [0, 0.2, 0.4],
        "learning_rate": [0.0001, 0.0005, 0.001],
        "batch_size": [16, 32, 64],
        "momentum": [0]
    }

    logger.info("Initializing Neural Network ...")

    X, y = nn.create_dataset(args.training_file)
    validation_benchmark = BenchmarkReader().read_benchmark(args.validation_file)
    X_val, y_val, entity_index = nn.create_dataset(args.validation_file, return_entity_index=True)

    # Generate parameter combinations as dictionaries
    keys = parameters.keys()
    param_combinations = [dict(zip(keys, values)) for values in product(*parameters.values())]
    params_with_hit_rate = {}

    # Loop through the combinations
    best_hit_rate = 0
    best_params = None
    for params in param_combinations:
        logger.info(f"{Colors.BOLD}Testing parameter combination{Colors.END}")
        print_parameters([params], parameters.keys())
        nn.initialize_model(hidden_layer_sizes=params["hidden_layer_size"],
                            hidden_layers=1,
                            dropout=params["dropout"],
                            activation=params["activation"])
        nn.train(X,
                 y,
                 learning_rate=params["learning_rate"],
                 batch_size=params["batch_size"],
                 optimizer=params["optimizer"],
                 momentum=params["momentum"],
                 X_val=X_val,
                 y_val=y_val,
                 entity_index=entity_index,
                 val_benchmark=validation_benchmark,
                 patience=5)

        evaluation_results = evaluate(nn.predict, validation_benchmark, [MetricName.HIT_RATE_AT_1])
        hit_rate = evaluation_results[MetricName.HIT_RATE_AT_1]
        logger.info(f"Accuracy @ 1: {hit_rate:.2f}")
        params_with_hit_rate[tuple(params.items())] = hit_rate
        if hit_rate > best_hit_rate:
            nn.save_model(args.save_model)
            best_hit_rate = hit_rate
            best_params = params
            print(f"{Colors.BOLD}New best model found with accuracy @ 1: {hit_rate:.2f}{Colors.END}")
        top_3_params_and_accuracy = sorted(params_with_hit_rate.items(), key=lambda x: x[1], reverse=True)[:3]
        top_3_params = [dict(p[0]) for p in top_3_params_and_accuracy]
        top_3_accuracies = [p[1] for p in top_3_params_and_accuracy]
        print(f"Currently best three parameter combinations:")
        print_parameters(top_3_params, parameters.keys(), top_k=3)
        print("\t", end="")
        for acc in top_3_accuracies:
            print(f"\t{acc:.2f}", end="")
        print()

    logger.info(f"Best model found with accuracy @ 1: {best_hit_rate:.2f} for parameters:")
    print_parameters([best_params], parameters.keys())

    if args.output_file:
        with open(args.output_file, "w") as f:
            for params, hit_rate in sorted(params_with_hit_rate.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{params}: {hit_rate}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    parser.add_argument("-train", "--training_file", type=str, required=True,
                        help="File that contains the training dataset.")
    parser.add_argument("-val", "--validation_file", type=str, required=True,
                        help="File containing the validation dataset. Relevant for the neural network model only.")
    parser.add_argument("--save_model", type=str, required=True, help="File to which to save the model.")
    parser.add_argument("--output_file", type=str, help="File to which to write the parameters sorted by hit rate to.")

    logger = log.setup_logger(stdout_level=logging.INFO)

    main(parser.parse_args())
