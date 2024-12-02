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

def main(args):
    entity_db = EntityDatabase()
    entity_db.load_instance_of_mapping()
    entity_db.load_subclass_of_mapping()
    entity_db.load_entity_to_name()
    entity_db.load_entity_to_description()
    nn = NeuralTypePredictor(entity_db)

    parameters = {
        "hidden_layer_size": [256, 512],
        "activation": ["sigmoid"],
        "hidden_layers": [1, 2],
        "dropout": [0.2, 0.4, 0.6],
        "learning_rate_init": [0.001, 0.01, 0.1],
        "batch_size": [16, 32, 64],
        "optimizer": ["SGD"],
        "momentum": [0, 0.3, 0.6]
    }

    logger.info("Initializing Neural Network ...")

    X, y = nn.create_dataset(args.training_file)
    validation_benchmark = BenchmarkReader().read_benchmark(args.validation_file)
    X_val, y_val, entity_index = nn.create_dataset(args.validation_file, return_entity_index=True)

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
        nn.train(X,
                 y,
                 learning_rate=params["learning_rate_init"],
                 batch_size=params["batch_size"],
                 optimizer=params["optimizer"],
                 momentum=params["momentum"],
                 X_val=X_val,
                 y_val=y_val,
                 entity_index=entity_index,
                 val_benchmark=validation_benchmark)

        evaluation_results = evaluate(nn.predict, validation_benchmark, [MetricName.HIT_RATE_AT_1])
        hit_rate = evaluation_results[MetricName.HIT_RATE_AT_1]
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
