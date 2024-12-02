import argparse
import logging
import sys
from itertools import product

sys.path.append(".")

from src.models.entity_database import EntityDatabase
from src.type_computation.neural_network import NeuralTypePredictor
from src.evaluation.benchmark_reader import BenchmarkReader
from src.evaluation.metrics import Metrics
from src.utils import log


def evaluate(scoring_function, benchmark):
    hit_rate_at_1 = []
    for entity_id in benchmark:
        result_types = scoring_function(entity_id)
        if result_types is None or len(result_types) == 0 or result_types[0] is None:
            result_types = ([None])
        elif type(result_types[0]) is tuple:
            result_types = [r[1] for r in result_types]  # Get only the type ids, not the scores
        hit_rate_at_1.append(Metrics.hit_rate_at_k(result_types, benchmark[entity_id], 1))

    mean_hit_rate_at_1 = sum(hit_rate_at_1) / len(hit_rate_at_1)

    return mean_hit_rate_at_1

def main(args):
    entity_db = EntityDatabase()
    entity_db.load_instance_of_mapping()
    entity_db.load_subclass_of_mapping()
    entity_db.load_entity_to_name()
    entity_db.load_entity_to_description()
    nn = NeuralTypePredictor(entity_db)
    validation_benchmark = BenchmarkReader().read_benchmark(args.validation_file)

    parameters = {
        "hidden_layer_size": [256, 512, 1024],
        "activation": ["relu", "tanh", "sigmoid"],
        "hidden_layers": [1, 2],
        "dropout": [0.1, 0.2, 0.3, 0.4],
        "learning_rate_init": [0.001, 0.01, 0.1],
        "batch_size": [16, 32, 64],
        "optimizer": ["SGD"],
        "momentum": [0, 0.3, 0.6]
    }

    parameters = {
        "hidden_layer_size": [256, 512],
        "activation": ["sigmoid"],
        "hidden_layers": [1, 2],
        "dropout": [0.2, 0.4, 0.6],
        "learning_rate_init": [0.001, 0.01, 0.1],
        "batch_size": [16, 32],
        "optimizer": ["SGD"],
        "momentum": [0, 0.3, 0.6]
    }


    parameters = {
        "hidden_layer_size": [512],
        "activation": ["sigmoid"],
        "hidden_layers": [1],
        "dropout": [0.2],
        "learning_rate_init": [0.01, 0.1],
        "batch_size": [16,],
        "optimizer": ["SGD"],
        "momentum": [0, 0.3]
    }

    logger.info("Initializing Neural Network ...")

    X, y = nn.create_dataset(args.training_file)
    X_val, y_val = nn.create_dataset(args.validation_file)

    # Generate parameter combinations as dictionaries
    keys = parameters.keys()
    param_combinations = [dict(zip(keys, values)) for values in product(*parameters.values())]

    # Loop through the combinations
    best_hit_rate = 0
    for params in param_combinations:
        logger.info(f"Testing combination: {params}")
        nn.initialize_model(hidden_layer_sizes=params["hidden_layer_size"],
                            hidden_layers=params["hidden_layers"],
                            dropout=params["dropout"],
                            activation=params["activation"])
        nn.train(X, y, learning_rate=params["learning_rate_init"], batch_size=params["batch_size"], optimizer=params["optimizer"], momentum=params["momentum"], X_val=X_val, y_val=y_val)

        hit_rate = evaluate(nn.predict, validation_benchmark)
        if hit_rate > best_hit_rate:
            nn.save_model(args.save_model)
            best_hit_rate = hit_rate
            logger.info(f"New best model found with hit rate: {hit_rate:.2f}")
            logger.info(f"Parameters: {params}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    parser.add_argument("-train", "--training_file", type=str, required=True,
                        help="File that contains the training dataset.")
    parser.add_argument("-val", "--validation_file", type=str, required=True,
                        help="File containing the validation dataset. Relevant for the neural network model only.")
    parser.add_argument("--save_model", type=str, required=True, help="File to which to save the model.")

    logger = log.setup_logger(stdout_level=logging.INFO)

    main(parser.parse_args())
