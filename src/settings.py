import logging
import os

logger = logging.getLogger("main." + __name__.split(".")[-1])


_DATA_DIRECTORIES = [
    "./data/",
    "/data/"
]
DATA_DIRECTORY = None
for directory in _DATA_DIRECTORIES:
    if os.path.exists(directory):
        DATA_DIRECTORY = directory
        break
if DATA_DIRECTORY is None:
    logger.error("Could not find the data directory.")
    exit(1)

ENTITY_TO_INSTANCE_OF_FILE = DATA_DIRECTORY + "wikidata_mappings/qid_to_p31.tsv"
ENTITY_TO_SUBCLASS_OF_FILE = DATA_DIRECTORY + "wikidata_mappings/qid_to_p279.tsv"
ENTITY_TO_NAME_FILE = DATA_DIRECTORY + "wikidata_mappings/qid_to_label.tsv"
ENTITY_TO_POPULARITY_FILE = DATA_DIRECTORY + "wikidata_mappings/qid_to_sitelinks.tsv"
ENTITY_TO_DESCRIPTION_FILE = DATA_DIRECTORY + "wikidata_mappings/qid_to_description.tsv"

TYPE_FREQUENCY_FILE = DATA_DIRECTORY + "type_frequencies.tsv"
TYPE_POPULARITY_FILE = DATA_DIRECTORY + "type_popularities.tsv"
AVG_TYPE_POPULARITY_FILE = DATA_DIRECTORY + "average_type_popularities.tsv"
ALL_ENTITY_TYPES = DATA_DIRECTORY + "all_entity_types.tsv"
TYPE_VARIANCE_FILE = DATA_DIRECTORY + "type_variance.tsv"

PROMINENT_TYPE_FILE = DATA_DIRECTORY + "prominent_types.tsv"
