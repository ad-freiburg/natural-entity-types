import sys
import time
import spacy
import random
import argparse
from collections import defaultdict

sys.path.append(".")


from src.models.entity_database import EntityDatabase
from src.utils import log


def main(args):
    entity_db = EntityDatabase()
    entity_db.load_entity_descriptions()
    entity_db.load_entity_to_name()
    entity_db.load_instance_of_mapping()
    entity_db.load_subclass_of_mapping()

    nlp = spacy.load("en_core_web_lg")
    training_data = []
    type_counter = defaultdict(int)
    indices = list(range(len(entity_db.entity_descriptions)))
    random.shuffle(indices)
    k = 0
    start = time.time()
    while k < len(entity_db.entity_descriptions):
        entity_index = indices[k]
        entity_id, description = entity_db.entity_descriptions[entity_index]
        entity_types = entity_db.get_entity_types(entity_id)
        entity_type_names = {entity_db.get_entity_name(t).lower(): t for t in entity_types if t and entity_db.get_entity_name(t)}
        doc = nlp(description)
        potential_gt_types = []
        if args.verbose:
            print(f"{entity_db.get_entity_name(entity_id)}: \n\t", end="")
        for i, tok in enumerate(doc):
            if args.verbose:
                print(f"{tok}", end=" ")
            if tok.dep_ == "ROOT":
                if args.verbose:
                    print(f"({tok.dep_})", end=" ")
                if tok.text.lower() in entity_type_names:
                    potential_gt_types.append(entity_type_names[tok.text.lower()])
                j = i - 1
                possible_type_name = tok.text.lower()
                while j > 0:
                    possible_type_name = doc[j].text.lower() + " " + possible_type_name
                    if possible_type_name in entity_type_names:
                        potential_gt_types.append(entity_type_names[possible_type_name])
                    j -= 1
        if args.verbose:
            print()
        if potential_gt_types and type_counter[tuple(potential_gt_types)] < 0.1 * args.num_samples:
            training_data.append((entity_id, potential_gt_types))
            type_counter[tuple(potential_gt_types)] += 1
            if args.verbose:
                print(f"\tPotential types: {', '.join([entity_db.get_entity_name(t) for t in potential_gt_types if t])}")
            if len(training_data) % 100 == 0:
                print(f"\r{len(training_data)} training samples generated in {time.time() - start} s.",  end='')
        if len(training_data) >= args.num_samples:
            break
        k += 1

    with open(args.output_file, "w", encoding="utf8") as output_file:
        for entry in training_data:
            entity_id = entry[0]
            potential_types = entry[1]
            output_file.write(f"{entity_id}\t")
            for i, t in enumerate(potential_types):
                output_file.write(f"{t}")
                if i < len(potential_types) - 1:
                    output_file.write(" ")
            output_file.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)

    parser.add_argument("-o", "--output_file", type=str,
                        help="File to which to write the evaluation results to.")
    parser.add_argument("-n", "--num_samples", type=int,
                        help="Number of samples to write to the training dataset.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print additional information while generating the data.")

    args = parser.parse_args()
    logger = log.setup_logger()

    main(args)
