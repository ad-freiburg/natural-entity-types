import argparse
import sys

sys.path.append(".")

import src.utils.log as log
from src import settings
from src.models.entity_database import EntityDatabase


def main(args):
    entity_db = EntityDatabase()
    entity_db.load_instance_of_mapping()
    entity_db.load_subclass_of_mapping()
    entity_db.load_entities()

    logger.info(f"Writing entity types to {args.output_file} ...")
    with open(args.output_file, "w", encoding="utf8") as output_file:
        for i, entity_id in enumerate(entity_db.entities):
            types = entity_db.get_entity_types(entity_id)
            for t in types:
                output_file.write(f"{entity_id}\t{t}\n")
            if i % 10000 == 0:
                print(f"\rProcessing entity no. {i}", end="")
    print()
    logger.info(f"Wrote {len(entity_db.entities)} entity types to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__)

    parser.add_argument("-o", "--output_file", type=str, default=settings.ALL_ENTITY_TYPES,
                        help="File to which to write the entity types to.")

    logger = log.setup_logger()

    main(parser.parse_args())
