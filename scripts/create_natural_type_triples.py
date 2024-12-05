import argparse
import logging
import sys
import time

sys.path.append(".")

from src.utils import log
from src.type_computation.neural_network import NeuralTypePredictor
from src.models.entity_database import EntityDatabase


def get_qids_from_file(file_path):
    with open(file_path, "r") as in_file:
        for line in in_file:
            # Make batch predictions to speed up the process
            qid = line.strip().split("\t")[0]
            yield qid


def get_qids_from_entity_db(entity_db):
    for qid in entity_db.typed_entities:
        yield qid


def process_batch(batch, nn, out_file):
    # Create a dataset from the collected QIDs
    X_batch, idx_to_ent_type_pair = nn.create_dataset_from_qids(batch)

    # Predict a score for each entity - candidate type pair in the dataset
    y_pred = nn.predict_batch(X_batch)

    # For each entity in the dataset predict the type with the highest score
    max_for_entity = 0, None
    current_entity = None
    for i, predicted_score in enumerate(y_pred):
        entity, typ = idx_to_ent_type_pair[i]
        if current_entity != entity:
            if current_entity is not None:
                out_file.write(f"wd:{current_entity} wdt:P31279 wd:{max_for_entity[1]} .\n")
            current_entity = entity
            max_for_entity = 0, None
        if predicted_score > max_for_entity[0]:
            max_for_entity = predicted_score, typ
    if current_entity is not None:
        out_file.write(f"wd:{current_entity} wdt:P31279 wd:{max_for_entity[1]} .\n")

def main(args):
    logger.info("Initializing Neural Network ...")
    entity_db = EntityDatabase()
    entity_db.load_instance_of_mapping()
    entity_db.load_subclass_of_mapping()
    entity_db.load_entity_to_name()
    entity_db.load_entity_to_description()
    entity_db.load_all_typed_entities(as_dictionary=True)

    nn = NeuralTypePredictor(entity_db)
    nn.load_model(args.load_model)

    batch_count = 0
    logger.setLevel(logging.WARNING)
    with open (args.output_file, "w", encoding="utf8") as out_file:
        qid_iterator = get_qids_from_file(args.input_file) if args.input_file else get_qids_from_entity_db(entity_db)
        batch = []
        start = time.time()
        for qid in qid_iterator:
            batch.append(qid)
            if len(batch) >= args.batch_size:
                batch_count += 1
                process_batch(batch, nn, out_file)
                print(f"\rProcessed {args.batch_size * batch_count} entities in {time.time() - start:.2f} s. "
                      f"Time per batch: {(time.time() - start) / batch_count:.2f} s", end="")

                # Start a new batch
                batch = []

                if batch_count >= 3:
                    break

        if batch:
            # Process the last incomplete batch
            process_batch(batch, nn, out_file)
            print(f"\rProcessed {(args.batch_size * batch_count) + len(batch)} entities in {time.time() - start:.2f} s. "
                  f"Time per batch: {(time.time() - start) / batch_count:.2f} s.", end="")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    parser.add_argument("--load_model", type=str, help="File from which to load the model.")
    parser.add_argument("-i", "--input_file", type=str,
                        help="Input file with one QID per line. A type is predicted for each QID. "
                             "If no input file is given, QLever is queried for all Wikidata items "
                             "which are subject of a triple.")
    parser.add_argument("-o", "--output_file", type=str,
                        help="Output file to which to write the triples.")
    parser.add_argument("--batch_size", type=int, default=10000,
                        help="Number of entities to put in one batch.")

    logger = log.setup_logger()

    main(parser.parse_args())
