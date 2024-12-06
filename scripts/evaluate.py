import sys
import argparse

sys.path.append(".")

from src.utils import log
from src.evaluation.benchmark_reader import BenchmarkReader
from src.evaluation.metrics import MetricName
from src.models.entity_database import EntityDatabase
from src.type_computation.model_names import ModelNames
from src.type_computation.neural_network import NeuralTypePredictor
from src.evaluation.evaluation import evaluate


def evaluate_method(scoring_function, benchmark, entity_db, verbose=False):
    metrics = [MetricName.HIT_RATE_AT_1, MetricName.HIT_RATE_AT_3, MetricName.HIT_RATE_AT_5,
               MetricName.HIT_RATE_AT_10, MetricName.MRR]

    evaluation_results = evaluate(scoring_function, benchmark, metrics, entity_db, verbose)

    for metric in evaluation_results:
        print(f"{metric.value}: {evaluation_results[metric]:.2f}")


def main(args):
    entity_db = EntityDatabase()
    entity_db.load_entity_to_name()

    # Initialize models
    predict_methods = []
    if ModelNames.MANUAL_SCORING.value in args.models:
        logger.info("Initializing manual type scorer...")
        from src.type_computation.prominent_type_computer import ProminentTypeComputer
        type_computer = ProminentTypeComputer(None, entity_db=entity_db)
        predict_methods.append((type_computer.compute_entity_score, ModelNames.MANUAL_SCORING.value))
    if ModelNames.GRADIENT_BOOST_REGRESSOR.value in args.models:
        logger.info("Initializing gradient boost regression model ...")
        from src.type_computation.gradient_boost_regressor import GradientBoostRegressor
        gb = GradientBoostRegressor(entity_db=entity_db)
        if args.load_model:
            gb.load_model(args.load_model)
        else:
            X, y = gb.create_dataset(args.training_file)
            gb.train(X, y)
        predict_methods.append((gb.predict, ModelNames.GRADIENT_BOOST_REGRESSOR.value))
        if args.save_model:
            gb.save_model(args.save_model)
    if ModelNames.GPT.value in args.models:
        logger.info("Initializing GPT ...")
        from src.type_computation.gpt import GPT
        gpt = GPT(entity_db, model="gpt-4o")
        predict_methods.append((gpt.predict, ModelNames.GPT.value))
    if ModelNames.NEURAL_NETWORK.value in args.models:
        logger.info("Initializing Neural Network ...")
        nn = NeuralTypePredictor(entity_db)
        if args.load_model:
            nn.load_model(args.load_model)
        else:
            nn.initialize_model(hidden_layer_sizes=512,
                                hidden_layers=1,
                                dropout=0.4,
                                activation="sigmoid")
            X, y = nn.create_dataset(args.training_file)
            validation_benchmark, X_val, y_val, entity_index = None, None, None, None
            if args.validation_file:
                validation_benchmark = BenchmarkReader().read_benchmark(args.validation_file)
                X_val, y_val, entity_index = nn.create_dataset(args.validation_file, return_entity_index=True)

            nn.train(X, y, X_val=X_val, y_val=y_val, batch_size=64, learning_rate=0.0001, optimizer="adam", entity_index=entity_index, val_benchmark=validation_benchmark)
        predict_methods.append((nn.predict, ModelNames.NEURAL_NETWORK.value))
        if args.save_model:
            nn.save_model(args.save_model)
    if ModelNames.ORACLE.value in args.models:
        logger.info("Initializing Oracle ...")
        from src.type_computation.oracle import Oracle
        oracle = Oracle(entity_db)
        predict_methods.append((oracle.predict, ModelNames.ORACLE.value))

    # Evaluate all models on all benchmarks
    for benchmark_file in args.benchmark_files:
        print()
        print(f"Evaluating benchmark {benchmark_file}")
        benchmark = BenchmarkReader.read_benchmark(benchmark_file)
        for predict_method, model_name in predict_methods:
            print(f"Evaluated model: {model_name}")
            print()
            if model_name == ModelNames.ORACLE.value:
                oracle.set_benchmark(benchmark)
            evaluate_method(predict_method, benchmark, entity_db, verbose=args.verbose)
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    parser.add_argument("-m", "--models", type=str, nargs="+", required=True, choices=[f.value for f in ModelNames],
                        help="Names of the models that will be evaluated.")
    parser.add_argument("--save_model", type=str, help="File to which to save the model.")
    parser.add_argument("--load_model", type=str, help="File from which to load the model.")
    parser.add_argument("-b", "--benchmark_files", type=str, required=True, nargs='+',
                        help="File that contains the benchmark.")
    parser.add_argument("-train", "--training_file", type=str,
                        help="File that contains the training dataset.")
    parser.add_argument("-val", "--validation_file", type=str,
                        help="File containing the validation dataset. Relevant for the neural network model only.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print details about each evaluated benchmark entity.")

    args = parser.parse_args()
    logger = log.setup_logger()

    if (not args.training_file and not args.load_model and
            (ModelNames.GRADIENT_BOOST_REGRESSOR.value in args.models or ModelNames.NEURAL_NETWORK in args.models)):
        logger.info("The model you selected requires that you provide a training file via the -train option or load a "
                    "pre-trained model using the --load_model option.")
        sys.exit(1)

    main(args)
