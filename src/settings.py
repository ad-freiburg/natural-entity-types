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

# Plain Wikidata mappings downloaded via QLever
ENTITY_TO_INSTANCE_OF_FILE = DATA_DIRECTORY + "wikidata_mappings/qid_to_p31.tsv"
ENTITY_TO_SUBCLASS_OF_FILE = DATA_DIRECTORY + "wikidata_mappings/qid_to_p279.tsv"
ENTITY_TO_NAME_FILE = DATA_DIRECTORY + "wikidata_mappings/qid_to_label.tsv"
ENTITY_TO_POPULARITY_FILE = DATA_DIRECTORY + "wikidata_mappings/qid_to_sitelinks.tsv"
ENTITY_TO_DESCRIPTION_FILE = DATA_DIRECTORY + "wikidata_mappings/qid_to_description.tsv"

# Databases
ENTITY_TO_DESCRIPTION_DB = DATA_DIRECTORY + "wikidata_mappings/qid_to_description.db"
ENTITY_TO_LABEL_DB = DATA_DIRECTORY + "wikidata_mappings/qid_to_label.db"
ENTITY_TO_SITELINKS_DB = DATA_DIRECTORY + "wikidata_mappings/qid_to_sitelinks.db"
ENTITY_TO_P31_DB = DATA_DIRECTORY + "wikidata_mappings/qid_to_p31.db"
ENTITY_TO_P279_DB = DATA_DIRECTORY + "wikidata_mappings/qid_to_p279.db"

# Computed feature mappings
TYPE_FREQUENCY_FILE = DATA_DIRECTORY + "computed_mappings/type_frequencies.tsv"
TYPE_POPULARITY_FILE = DATA_DIRECTORY + "computed_mappings/type_popularities.tsv"
ALL_ENTITY_TYPES = DATA_DIRECTORY + "computed_mappings/all_entity_types.tsv"
TYPE_VARIANCE_FILE = DATA_DIRECTORY + "computed_mappings/type_variance.tsv"
PROMINENT_TYPE_FILE = DATA_DIRECTORY + "computed_mappings/prominent_types.tsv"

PREDICATES_DIRECTORY = DATA_DIRECTORY + "predicates/"
PREDICATE_VARIANCES_DIRECTORY = DATA_DIRECTORY + "predicate_variances/"

TMP_FORKSERVER_CONFIG_FILE = DATA_DIRECTORY + "tmp_forkserver_config.json"