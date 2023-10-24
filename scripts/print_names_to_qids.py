import argparse
import sys

sys.path.append(".")

from src.models.entity_database import EntityDatabase
import src.utils.log as log


def main(args):
    entity_db = EntityDatabase()
    entity_db.load_entity_to_name()

    with open(args.input_file, "r", encoding="utf8") as file:
        for line in file:
            lst = line.strip("\n").split("\t")
            for i, el in enumerate(lst):
                if not el:
                    continue
                separator = "" if i == 0 else "\t"
                entity_name = entity_db.get_entity_name(el)
                name_suffix = " (" + entity_name + ")" if entity_name else ""
                print(f"{separator}{el}{name_suffix}", end="")
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__)

    parser.add_argument("-i", "--input_file", type=str, required=True,
                        help="Input TSV file with QIDs.")

    logger = log.setup_logger()

    main(parser.parse_args())
