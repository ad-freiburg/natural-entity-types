import argparse
import sys


sys.path.append(".")

from src.type_computation.prominent_type_computer import ProminentTypeComputer
import src.utils.log as log


def main(args):
    type_computer = ProminentTypeComputer(args.input_files, args.output_file)
    if args.interactive:
        import re
        type_computer.entity_db.load_name_to_entity()
        type_computer.entity_db.load_entity_popularity()
        while True:
            identifier = input("Enter a QID or an entity name: ")
            if re.match(r"Q[1-9][0-9]*", identifier):
                type_computer.compute_entity_score(identifier, verbose=True)
            else:
                entity_id = type_computer.entity_db.get_entity_by_name(identifier)
                if entity_id:
                    type_computer.compute_entity_score(entity_id, verbose=True)
    elif args.in_order:
        prominent_types = type_computer.compute_all_entity_scores(args.num_entities)
        if args.output_file:
            type_computer.write_entity_scores(prominent_types, args.output_file)
    else:
        prominent_types = type_computer.compute_random_entity_scores(args.num_entities)
        if args.output_file:
            type_computer.write_entity_scores(prominent_types, args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__)

    parser.add_argument("-i", "--input_files", type=str, required=True, nargs='+',
                        help="File that contains the predicate variance scores")
    parser.add_argument("-o", "--output_file", type=str,
                        help="File to which to write the entity to prominent type mapping to.")
    parser.add_argument("-n", "--num_entities", type=int, default=100,
                        help="Number of entities to process.")
    parser.add_argument("-a", "--interactive", action="store_true",
                        help="Ask the user for an entity and compute the scores for the types of this entity.")
    parser.add_argument("--in_order", action="store_true",
                        help="Go through the entities in order of their QIDs instead of randomly.")

    logger = log.setup_logger()

    main(parser.parse_args())
