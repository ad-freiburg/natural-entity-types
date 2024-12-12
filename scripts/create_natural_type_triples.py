import argparse
import sys
import time
import multiprocessing
import json

sys.path.append(".")

from src.utils import log
from src.models.entity_database import EntityDatabase
from src import settings

# Importing the neural network has to happen here, otherwise a neural network is not defined error is thrown.
# In order to not load several GB of mappings twice, only do a fake loading such that the name is imported,
# but no mappings are loaded
config = {"no_loading": True}
with open(settings.TMP_FORKSERVER_CONFIG_FILE, "w", encoding="utf8") as config_file:
    json.dump(config, config_file)
from src.type_computation.forkserver_neural_network import nn

CHUNK_SIZE = 10
MAX_TASKS_PER_CHILD = 10

LABEL_PROPERTIES = ["@prefix wikibase: <http://wikiba.se/ontology#> .",
                    "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
                    "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
                    "@prefix schema: <http://schema.org/> .",
                    "@prefix wd: <http://www.wikidata.org/entity/> .",
                    "@prefix wdt: <http://www.wikidata.org/prop/direct/> .",
                    "wd:P31279 wikibase:directClaim wdt:P31279 .",
                    "wd:P31279 rdfs:label \"natural type\"@en .",
                    "wd:P31279 schema:name \"natural type\"@en .",
                    "wd:P31279 schema:description \"the most natural type of an entity\"@en .",
                    "wd:P31279 rdf:type wikibase:Property ."]

def get_qids_from_file(file_path, batch_size, n=-1):
    entities = []
    i = 0
    with open(file_path, "r") as in_file:
        for line in in_file:
            # Make batch predictions to speed up the process
            e = line.strip().split("\t")[0]
            entities.append(e)
            i += 1
            if len(entities) >= batch_size:
                yield entities
                entities = []
            if i == n:
                break
        if entities:
            yield entities


def get_qids_from_entity_db(entity_db, batch_size, n=-1):
    entities = []
    i = 0
    for e in entity_db.typed_entities:
        entities.append(e)
        i += 1
        if len(entities) >= batch_size:
            yield entities
            entities = []
        if i == n:
            break
    if entities:
        yield entities


def process_batch(batch):
    # Create a dataset from the collected QIDs
    X_batch, idx_to_ent_type_pair = nn.create_dataset_from_qids(batch)

    # Predict a score for each entity - candidate type pair in the dataset
    y_pred = nn.predict_batch(X_batch)

    # For each entity in the dataset predict the type with the highest score
    max_for_entity = 0, None
    current_entity = None
    entities_with_types = []
    for i, predicted_score in enumerate(y_pred):
        entity, typ = idx_to_ent_type_pair[i]
        if current_entity != entity:
            if current_entity is not None:
                entities_with_types.append((current_entity, max_for_entity[1]))
            current_entity = entity
            max_for_entity = 0, None
        if predicted_score > max_for_entity[0]:
            max_for_entity = predicted_score, typ
    if current_entity is not None:
        entities_with_types.append((current_entity, max_for_entity[1]))
    return entities_with_types


def main(args):
    if args.input_file:
        entity_iterator = get_qids_from_file(args.input_file, args.batch_size, args.num_entities)
    else:
        entity_db = EntityDatabase()
        entity_db.load_all_typed_entities()
        entity_iterator = get_qids_from_entity_db(entity_db, args.batch_size, args.num_entities)

    with open (args.output_file, "w", encoding="utf8") as out_file:
        # First write the "natural type" property triples to the output file
        for prop in LABEL_PROPERTIES:
            out_file.write(f"{prop}\n")

        # Predict types for all entities using multiprocessing
        multiprocessing.set_start_method('forkserver')
        multiprocessing.set_forkserver_preload(["src.type_computation.forkserver_neural_network"])
        i = 0
        start = time.time()
        with multiprocessing.Pool(processes=args.multiprocessing, maxtasksperchild=MAX_TASKS_PER_CHILD) as executor:
            logger.info(f"Start predicting entity types using {args.multiprocessing} processes.")
            for entities_with_types in executor.imap(process_batch, entity_iterator, chunksize=CHUNK_SIZE):
                for entity, typ in entities_with_types:
                    out_file.write(f"wd:{entity} wdt:P31279 wd:{typ} .\n")
                    i += 1
                    if i % 1000 == 0:
                        total_time = time.time() - start
                        avg_time = total_time / i
                        print(f"\r{i} entities, {avg_time*1000:.3f} s per 1000 entities, {int(total_time)} s total time.", end='')
        print()
    logger.info(f"Wrote {i} triples to {args.output_file} in {time.time() - start:.2f}s.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    parser.add_argument("--load_model", type=str, required=True,
                        help="File from which to load the model.")
    parser.add_argument("-i", "--input_file", type=str,
                        help="Input file with one QID per line. A type is predicted for each QID. "
                             "If no input file is given, the types for all entities with an instance-of "
                             "or subclass-of relation in the Wikidata mappings are predicted.")
    parser.add_argument("-o", "--output_file", type=str,
                        help="Output file to which to write the triples.")
    parser.add_argument("--batch_size", type=int, default=1000,
                        help="Number of entities to put in one batch.")
    parser.add_argument("-n", "--num_entities", type=int, default=-1,
                        help="Number of entities to process.")
    parser.add_argument("-m", "--multiprocessing", type=int, default=8,
                        help="Number of processes to use for dataset creation.")

    logger = log.setup_logger()

    args = parser.parse_args()

    # Write command line arguments to temporary config file which is then read by the forkserver_neural_network module.
    # This is not mega pretty, but I don't see a better solution where the user can still use command line arguments to
    # configure the neural network.
    config = {"model_path": args.load_model}
    with open(settings.TMP_FORKSERVER_CONFIG_FILE, "w", encoding="utf8") as config_file:
        json.dump(config, config_file)

    main(args)
