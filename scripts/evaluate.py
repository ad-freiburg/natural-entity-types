import sys
import argparse

sys.path.append(".")

from src.utils import log
from src.evaluation.benchmark_reader import BenchmarkReader
from src.evaluation.metrics import Metrics
from src.models.entity_database import EntityDatabase
from src.type_computation.model_names import ModelNames
from src.type_computation.neural_network import NeuralTypePredictor


def evaluate(scoring_function, benchmark, entity_db, output_file=None, verbose=False):
    if output_file:
        output_file = open(output_file, "w", encoding="utf8")
    aps = []
    p_at_rs = []
    hit_rate_at_1 = []
    hit_rate_at_3 = []
    hit_rate_at_5 = []
    hit_rate_at_10 = []
    rrs = []
    for entity_id in benchmark:
        result_types = scoring_function(entity_id)
        if result_types is None or len(result_types) == 0 or result_types[0] is None:
            result_types = ([None])
        elif type(result_types[0]) is tuple:
            result_types = [r[1] for r in result_types]  # Get only the type ids, not the scores
        ap = Metrics.average_precision(result_types, benchmark[entity_id])
        p_at_r = Metrics.precision_at_k(result_types, benchmark[entity_id], len(benchmark[entity_id]))
        entity_name = entity_db.get_entity_name(entity_id)
        gt_entities = ", ".join([f"{entity_db.get_entity_name(t)} ({t})" for t in benchmark[entity_id]])
        predicted_entities = ", ".join([f"{entity_db.get_entity_name(t)} ({t})" for t in result_types[:6]]) + "..."
        if verbose:
            print(f"Average precision for \"{entity_name}\" ({entity_id}): {ap:.2f}.\n"
                  f"\tGround truth: {gt_entities}\n"
                  f"\tprediction: {predicted_entities}")

        aps.append(ap)
        p_at_rs.append(p_at_r)
        hit_rate_at_1.append(Metrics.hit_rate_at_k(result_types, benchmark[entity_id], 1))
        hit_rate_at_3.append(Metrics.hit_rate_at_k(result_types, benchmark[entity_id], 3))
        hit_rate_at_5.append(Metrics.hit_rate_at_k(result_types, benchmark[entity_id], 5))
        hit_rate_at_10.append(Metrics.hit_rate_at_k(result_types, benchmark[entity_id], 10))
        rrs.append(Metrics.mrr(result_types, benchmark[entity_id]))
        if output_file:
            output_file.write(f"Average precision for \"{entity_name}\" ({entity_id}): {ap:.2f}.\n"
                              f"\tGround truth: {gt_entities}\n"
                              f"\tprediction: {predicted_entities}\n")
    mean_ap = sum(aps) / len(aps)
    mean_p_at_r = sum(p_at_rs) / len(p_at_rs)
    mean_hit_rate_at_1 = sum(hit_rate_at_1) / len(hit_rate_at_1)
    mean_hit_rate_at_3 = sum(hit_rate_at_3) / len(hit_rate_at_3)
    mean_hit_rate_at_5 = sum(hit_rate_at_5) / len(hit_rate_at_5)
    mean_hit_rate_at_10 = sum(hit_rate_at_10) / len(hit_rate_at_10)
    mrr = sum(rrs) / len(rrs)
    print(f"MAP: {mean_ap:.2f}")
    print(f"MP @ R: {mean_p_at_r:.2f}")
    print(f"Hit rate at 1: {mean_hit_rate_at_1:.2f}")
    print(f"Hit rate at 3: {mean_hit_rate_at_3:.2f}")
    print(f"Hit rate at 5: {mean_hit_rate_at_5:.2f}")
    print(f"Hit rate at 10: {mean_hit_rate_at_10:.2f}")
    print(f"MRR: {mrr:.2f}")

    if output_file:
        output_file.write(f"MAP: {mean_ap:.2f}\n")
        output_file.write(f"MP @ R: {mean_p_at_r:.2f}\n")
        output_file.close()


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
            nn.initialize_model(512, 0.2)
            X, y = nn.create_dataset(args.training_file)
            X_val, y_val = None, None
            if args.validation_file:
                X_val, y_val = nn.create_dataset(args.validation_file)
            nn.train(X, y, X_val=X_val, y_val=y_val)
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
        print(f"***** Evaluating benchmark {benchmark_file} *****")
        benchmark = BenchmarkReader.read_benchmark(benchmark_file)
        for predict_method, model_name in predict_methods:
            print(f"***** Evaluating {model_name} *****")
            if model_name == ModelNames.ORACLE.value:
                oracle.set_benchmark(benchmark)
            evaluate(predict_method, benchmark, entity_db, args.output_file, verbose=args.verbose)


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
    parser.add_argument("-o", "--output_file", type=str,
                        help="File to which to write the evaluation results to.")
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
