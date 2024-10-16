import sys
import argparse


sys.path.append(".")

from src.utils import log
from src.evaluation.benchmark_reader import BenchmarkReader
from src.evaluation.metrics import Metrics
from src.models.entity_database import EntityDatabase
from src.type_computation.prominent_type_computer import ProminentTypeComputer
from src.type_computation.gradient_boost_regressor import GradientBoostRegressor
from src.type_computation.gpt import GPT
from src.type_computation.model_names import ModelNames
from src.type_computation.neural_network import NeuralTypePredictor


def evaluate(scoring_function, benchmark, entity_db, output_file=None):
    if output_file:
        output_file = open(output_file, "w", encoding="utf8")
    aps = []
    p_at_1s = []
    p_at_rs = []
    for entity_id in benchmark:
        result_types = scoring_function(entity_id)
        if result_types is None or result_types[0] is None:
            result_types = ([None])
        elif type(result_types[0]) is tuple:
            result_types = [r[1] for r in result_types]  # Get only the type ids, not the scores
        ap = Metrics.average_precision(result_types, benchmark[entity_id])
        p_at_1 = Metrics.precision_at_k(result_types, benchmark[entity_id], 1)
        p_at_r = Metrics.precision_at_k(result_types, benchmark[entity_id], len(benchmark[entity_id]))
        entity_name = entity_db.get_entity_name(entity_id)
        gt_entities = ", ".join([f"{entity_db.get_entity_name(t)} ({t})" for t in benchmark[entity_id]])
        predicted_entities = ", ".join([f"{entity_db.get_entity_name(t)} ({t})" for t in result_types[:10]]) + "..."
        print(f"Average precision for \"{entity_name}\" ({entity_id}): {ap:.2f}.\n"
              f"\tGround truth: {gt_entities}\n"
              f"\tprediction: {predicted_entities}")
        aps.append(ap)
        p_at_1s.append(p_at_1)
        p_at_rs.append(p_at_r)
        if output_file:
            output_file.write(f"Average precision for \"{entity_name}\" ({entity_id}): {ap:.2f}.\n"
                              f"\tGround truth: {gt_entities}\n"
                              f"\tprediction: {predicted_entities}\n")
    mean_ap = sum(aps) / len(aps)
    mean_p_at_1 = sum(p_at_1s) / len(p_at_1s)
    mean_p_at_r = sum(p_at_rs) / len(p_at_rs)
    print(f"Mean average precision: {mean_ap:.2f}")
    print(f"Mean precision at 1: {mean_p_at_1:.2f}")
    print(f"Mean precision at R: {mean_p_at_r:.2f}")

    if output_file:
        output_file.write(f"Mean average precision: {mean_ap:.2f}\n")
        output_file.write(f"Mean precision at 1: {mean_p_at_1:.2f}\n")
        output_file.write(f"Mean precision at R: {mean_p_at_r:.2f}\n")
        output_file.close()


def main(args):
    entity_db = EntityDatabase()
    entity_db.load_entity_to_name()

    benchmark = BenchmarkReader.read_benchmark(args.benchmark_file)

    if ModelNames.MANUAL_SCORING.value in args.models:
        logger.info("Loading manual type scorer...")
        type_computer = ProminentTypeComputer(args.input_files, None, entity_db=entity_db)
        logger.info("Evaluating manual type scorer...")
        evaluate(type_computer.compute_entity_score, benchmark, entity_db, args.output_file)
    if ModelNames.GRADIENT_BOOST_REGRESSOR.value in args.models:
        logger.info("Loading gradient boost regression model ...")
        gb = GradientBoostRegressor(args.input_files, entity_db=entity_db)
        X, y = gb.create_dataset(args.training_file)
        gb.train(X, y)
        logger.info("Evaluating gradient boost regression model ...")
        evaluate(gb.predict, benchmark, entity_db, args.output_file)
    if ModelNames.GPT.value in args.models:
        logger.info("Loading GPT ...")
        gpt = GPT(entity_db)
        logger.info("Evaluating GPT ...")
        evaluate(gpt.predict, benchmark, entity_db, args.output_file)
    if ModelNames.NEURAL_NETWORK.value in args.models:
        logger.info("Loading Neural Network ...")
        nn = NeuralTypePredictor(args.input_files, entity_db)
        nn.initialize_model(8 + 300*2, 512, 0)
        X, y = nn.create_dataset(args.training_file)
        nn.train(X, y)
        evaluate(nn.predict, benchmark, entity_db, args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    parser.add_argument("-m", "--models", type=str, nargs="+", required=True, choices=[f.value for f in ModelNames],
                        help="Names of the models that will be evaluated.")
    parser.add_argument("-b", "--benchmark_file", type=str, required=True,
                        help="File that contains the benchmark.")
    parser.add_argument("-i", "--input_files", type=str, nargs='+',
                        help="File that contains the predicate variance scores")
    parser.add_argument("-train", "--training_file", type=str,
                        help="File that contains the training dataset.")
    parser.add_argument("-o", "--output_file", type=str,
                        help="File to which to write the evaluation results to.")

    args = parser.parse_args()
    logger = log.setup_logger()

    if not args.input_files and (ModelNames.MANUAL_SCORING.value in args.models or
                                 ModelNames.GRADIENT_BOOST_REGRESSOR.value in args.models):
        logger.info("The model you selected requires that you provide predicate variance score files via the -i option.")
        sys.exit(1)
    if not args.training_file and ModelNames.GRADIENT_BOOST_REGRESSOR.value in args.models:
        logger.info("The model you selected requires that you provide a training file via the -train option.")
        sys.exit(1)

    main(args)
