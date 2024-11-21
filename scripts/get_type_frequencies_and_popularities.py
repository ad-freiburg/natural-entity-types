import sys
from collections import Counter

sys.path.append(".")

import src.utils.log as log
from src import settings
from src.models.entity_database import EntityDatabase
from src.utils.utils import create_directories_for_file


def main():
    entity_db = EntityDatabase()
    entity_db.load_instance_of_mapping(as_dictionary=True)  # Load mapping as a dictionary for faster access
    entity_db.load_subclass_of_mapping(as_dictionary=True)
    entity_db.load_entities()
    entity_db.load_entity_popularity(as_dictionary=True)


    type_frequencies = Counter()
    type_popularities = {}
    num_entities = len(entity_db.entities)
    for i, entity_id in enumerate(entity_db.entities):
        types = entity_db.get_entity_types(entity_id)
        popularity = entity_db.get_entity_popularity(entity_id)
        for t in types:
            if t not in type_popularities:
                type_popularities[t] = 0
            type_popularities[t] += popularity
            type_frequencies[t] += 1
        if i % 10000 == 0:
            print(f"\rProcessed {i+1} entities of {num_entities} total entities", end="")
    print()

    type_frequency_file = settings.TYPE_FREQUENCY_FILE
    logger.info(f"Writing type frequencies to {type_frequency_file} ...")
    create_directories_for_file(type_frequency_file)
    with open(type_frequency_file, "w", encoding="utf8") as output_file:
        for entity_id, frequency in type_frequencies.most_common():
            output_file.write(f"{entity_id}\t{frequency}\n")
    logger.info(f"Wrote {len(type_frequencies)} type frequencies to {type_frequency_file}")

    type_popularity_file = settings.TYPE_POPULARITY_FILE
    logger.info(f"Writing accumulated type popularities to {type_popularity_file} ...")
    create_directories_for_file(type_popularity_file)
    with open(type_popularity_file, "w", encoding="utf8") as output_file:
        for entity_id, popularity in type_popularities.items():
            output_file.write(f"{entity_id}\t{popularity}\n")
    logger.info(f"Wrote {len(type_popularities)} accumulated type popularities to {type_popularity_file}")


if __name__ == "__main__":
    logger = log.setup_logger()
    main()
