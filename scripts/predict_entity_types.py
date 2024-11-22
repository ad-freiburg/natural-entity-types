import sys
import argparse

sys.path.append(".")

from src.utils import log
from src.models.entity_database import EntityDatabase
from src.type_computation.prominent_type_computer import ProminentTypeComputer
from src.type_computation.gradient_boost_regressor import GradientBoostRegressor
from src.type_computation.gpt import GPT
from src.type_computation.model_names import ModelNames


def predict(input_file, output_file, scoring_function, entity_db=None):
    with open(output_file, "w", encoding="utf8") as output_file:
        with open(input_file, "r", encoding="utf8") as input_file:
            for i, line in enumerate(input_file):
                lst = line.strip("\n").split("\t")
                entity_uri = lst[0]
                entity_id = entity_uri[entity_uri.rfind("/") + 1:-1]
                candidate_types = [t.strip() for t in lst[4].split(";")]
                candidate_types = [t[t.rfind("/") + 1:] for t in candidate_types]
                result_types = scoring_function(entity_id, candidate_types)
                if result_types:
                    if type(result_types[0]) is tuple:
                        result_types = [r[1] for r in result_types]  # Get only the type ids, not the scores
                    lst[4] = "http://www.wikidata.org/entity/" + result_types[0]

                output_file.write("\t".join(lst) + "\n")
                if (i + 1) % 10_000 == 0:
                    print(f"\rWrote {i + 1} entity types", end="")


def main(args):
    entity_db = EntityDatabase()
    entity_db.load_entity_to_name()

    if ModelNames.MANUAL_SCORING.value in args.model:
        logger.info("Loading manual type scorer...")
        type_computer = ProminentTypeComputer(None, entity_db=entity_db)
        logger.info("Predicting with manual type scorer...")
        predict(args.input_file, args.output_file, type_computer.compute_entity_score)
    elif ModelNames.GRADIENT_BOOST_REGRESSOR.value in args.model:
        logger.info("Loading gradient boost regression model ...")
        gb = GradientBoostRegressor(entity_db=entity_db)
        X, y = gb.create_dataset(args.training_file)
        gb.train(X, y)
        logger.info("Predicting with gradient boost regression model ...")
        predict(args.input_file, args.output_file, gb.predict)
    elif ModelNames.GPT.value in args.model:
        gpt = GPT(entity_db)
        logger.info("Predicting with GPT ...")
        predict(args.input_file, args.output_file, gpt.predict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    parser.add_argument("-m", "--model", type=str, required=True, choices=[f.value for f in ModelNames],
                        help="Names of the model that will be evaluated.")
    parser.add_argument("-i", "--input_file", type=str, required=True,
                        help="Input file with entities for which to predict types.")
    parser.add_argument("-o", "--output_file", type=str, required=True,
                        help="File to which to write the evaluation results to.")
    parser.add_argument("-train", "--training_file", type=str,
                        help="File that contains the training dataset.")

    args = parser.parse_args()
    logger = log.setup_logger()

    if not args.training_file and ModelNames.GRADIENT_BOOST_REGRESSOR.value in args.model:
        logger.info("The model you selected requires that you provide a training file via the -train option.")
        sys.exit(1)

    main(args)
