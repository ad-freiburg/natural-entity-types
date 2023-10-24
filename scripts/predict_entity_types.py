import sys
import argparse

sys.path.append(".")

from src.utils import log
from src.models.entity_database import EntityDatabase
from src.type_computation.prominent_type_computer import ProminentTypeComputer
from src.type_computation.gradient_boost_regressor import GradientBoostRegressor
from src.type_computation.gpt import GPT
from src.type_computation.model_names import ModelNames


def predict(input_file, scoring_function, entity_db, output_file=None):
    with open(output_file, "w", encoding="utf8") as output_file:
        with open(input_file, "r", encoding="utf8") as input_file:
            for line in input_file:
                lst = line.strip("\n").split("\t")
                entity_id = lst[0]
                result_types = scoring_function(entity_id)
                if type(result_types[0]) is tuple:
                    result_types = [r[1] for r in result_types]  # Get only the type ids, not the scores

                entity_name = entity_db.get_entity_name(entity_id)
                predicted_entities = ", ".join([f"{entity_db.get_entity_name(t)} ({t})" for t in result_types[:10]]) + "..."
                output_file.write(f"")


def main(args):
    entity_db = EntityDatabase()
    entity_db.load_entity_to_name()

    if ModelNames.MANUAL_SCORING.value in args.models:
        logger.info("Loading manual type scorer...")
        type_computer = ProminentTypeComputer(args.predicate_files, None, entity_db=entity_db)
        logger.info("Predicting with manual type scorer...")
        predict(args.input_file, type_computer.compute_entity_score, entity_db, args.output_file)
    if ModelNames.GRADIENT_BOOST_REGRESSOR.value in args.models:
        logger.info("Loading gradient boost regression model ...")
        gb = GradientBoostRegressor(args.predicate_files, entity_db=entity_db)
        X, y = gb.create_dataset(args.training_file)
        gb.train(X, y)
        logger.info("Predicting with gradient boost regression model ...")
        predict(args.input_file, gb.predict, entity_db, args.output_file)
    if ModelNames.GPT.value in args.models:
        gpt = GPT(entity_db)
        logger.info("Predicting with GPT ...")
        predict(args.input_file, gpt.predict, entity_db, args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    parser.add_argument("-m", "--models", type=str, nargs="+", required=True, choices=[f.value for f in ModelNames],
                        help="Names of the models that will be evaluated.")
    parser.add_argument("-p", "--predicate_files", type=str, nargs='+',
                        help="File that contains the predicate variance scores")
    parser.add_argument("-train", "--training_file", type=str,
                        help="File that contains the training dataset.")
    parser.add_argument("-i", "--input_file", type=str,
                        help="Input file with entities for which to predict types.")
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
