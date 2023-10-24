import argparse
import sys

sys.path.append(".")

from src.utils import log
from src.type_computation.gradient_boost_classifier import GradientBoostClassifier
from src.type_computation.gradient_boost_regressor import GradientBoostRegressor


def main(args):
    if args.model.lower() in ["c", "classifier"]:
        gb = GradientBoostClassifier(args.input_files)
    else:
        gb = GradientBoostRegressor(args.input_files)

    X, y = gb.create_dataset(args.training_file)
    gb.train(X, y)

    if args.test_file:
        gb.evaluate(args.test_file)
        gb.plot_feature_importance(args.test_file)
        gb.plot_learning_curve(args.test_file)

    while True:
        entity_id = input("Enter a QID: ").strip()
        entity_name = gb.entity_db.get_entity_name(entity_id)
        prediction_scores = gb.predict(entity_id)
        if prediction_scores:
            print(f"Predicted types for {entity_name} ({entity_id}):")
            for i, (_, predicted_type_id) in enumerate(prediction_scores[:5]):
                predicted_type_name = gb.entity_db.get_entity_name(predicted_type_id)
                print(f"{i+1}. {predicted_type_name} ({predicted_type_id})")
        else:
            print("Model couldn't make a prediction.")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__)

    parser.add_argument("-i", "--input_files", type=str, required=True, nargs='+',
                        help="File that contains the predicate variance scores")
    parser.add_argument("-train", "--training_file", type=str, required=True,
                        help="File containing the training dataset.")
    parser.add_argument("-test", "--test_file", type=str,
                        help="File containing the test dataset.")
    parser.add_argument("-m", "--model", type=str, default="r",
                        help="Gradient boost model, either c for classifier or r for regressor")
    logger = log.setup_logger()

    main(parser.parse_args())
