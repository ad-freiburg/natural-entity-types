import argparse
import sys

sys.path.append(".")

import src.utils.log as log
from src import settings
from src.models.entity_database import EntityDatabase


def main(args):
    entity_db = EntityDatabase()
    entity_db.load_entity_to_name()
    entity_db.load_accumulated_type_popularity()
    entity_db.load_type_frequency()

    logger.info(f"Writing average type popularities to {args.output_file} ...")
    with open(args.output_file, "w", encoding="utf8") as output_file:
        for i, (entity_id, popularity) in enumerate(sorted(entity_db.accumulated_type_popularity.items())):
            frequency = entity_db.get_type_frequency(entity_id)
            avg_popularity = popularity / frequency
            entity_name = entity_db.get_entity_name(entity_id)
            output_file.write(f"{entity_id}\t{entity_name}\t{avg_popularity}\n")
            if i % 10000 == 0:
                print(f"\rProcessing entity no. {i}", end="")
    print()

    logger.info(f"Wrote {i} average type popularities to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__)

    parser.add_argument("-o", "--output_file", type=str, default=settings.AVG_TYPE_POPULARITY_FILE,
                        help="File to which to write the average type popularities to.")

    logger = log.setup_logger()

    main(parser.parse_args())
