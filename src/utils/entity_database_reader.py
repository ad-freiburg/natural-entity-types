import dbm
import logging
from typing import Optional

from src import settings
from src.models.database import Database

logger = logging.getLogger("main." + __name__.split(".")[-1])


class EntityDatabaseReader:
    @staticmethod
    def read_instance_of_mapping(relevant_entities=None):
        filename = settings.ENTITY_TO_INSTANCE_OF_FILE
        logger.info("Loading instance-of mapping from %s ..." % filename)
        if relevant_entities:
            logger.info("Loading restricted to %d relevant entities." % len(relevant_entities))
        mapping = EntityDatabaseReader.read_item_to_qid_set_mapping(filename, relevant_entities)
        logger.info("-> %d instance-of mappings loaded." % len(mapping))
        return mapping

    @staticmethod
    def read_reverse_instance_of_mapping():
        filename = settings.ENTITY_TO_INSTANCE_OF_FILE
        logger.info("Loading reverse instance-of mapping from %s ..." % filename)
        instance_of_types = {}
        with open(filename, "r", encoding="utf8") as f:
            for line in f:
                entity_id, type_id = line.strip('\n').split('\t')
                if entity_id[0] != "Q":
                    continue
                if type_id not in instance_of_types:
                    instance_of_types[type_id] = set()
                instance_of_types[type_id].add(entity_id)
        logger.info(f"-> {len(instance_of_types)} reverse instance-of mappings loaded.")
        return instance_of_types

    @staticmethod
    def read_subclass_of_mapping(relevant_entities=None):
        filename = settings.ENTITY_TO_SUBCLASS_OF_FILE
        logger.info("Loading subclass-of mapping from %s" % filename)
        if relevant_entities:
            logger.info("Loading restricted to %d relevant entities." % len(relevant_entities))
        mapping = EntityDatabaseReader.read_item_to_qid_set_mapping(filename, relevant_entities)
        logger.info("-> %d subclass-of mappings loaded." % len(mapping))
        return mapping

    @staticmethod
    def read_reverse_subclass_of_mapping():
        filename = settings.ENTITY_TO_SUBCLASS_OF_FILE
        logger.info("Loading reverse subclass-of mapping from %s" % filename)
        reverse_subclass_of_mapping = {}
        with open(filename, "r", encoding="utf8") as f:
            for line in f:
                subj_id, obj_id = line.strip('\n').split('\t')
                if obj_id not in reverse_subclass_of_mapping:
                    reverse_subclass_of_mapping[obj_id] = []
                reverse_subclass_of_mapping[obj_id].append(subj_id)
        logger.info("-> %d reverse subclass-of mappings loaded." % len(reverse_subclass_of_mapping))
        return reverse_subclass_of_mapping

    @staticmethod
    def read_item_to_qid_set_mapping(mapping_file: str, relevant_items):
        mapping = {}
        with open(mapping_file, "r", encoding="utf8") as f:
            for line in f:
                key, value = line.strip('\n').split('\t')
                if not relevant_items or key in relevant_items:
                    # Could also be "unknown value" in Wikidata which yields sth like _:134b940e46468ab95602a542cefecb52
                    if value and value[0] == "Q":
                        if key not in mapping:
                            mapping[key] = set()
                        mapping[key].add(value)
        return mapping

    @staticmethod
    def read_name_mapping():
        filename = settings.ENTITY_TO_NAME_FILE
        logger.info("Loading name mapping from %s ..." % filename)
        mapping = {}
        with open(filename, "r", encoding="utf8") as f:
            for line in f:
                key, value = line.strip('\n').split('\t')
                mapping[key] = value
        logger.info("-> %d name mappings loaded." % len(mapping))
        return mapping

    @staticmethod
    def read_name_to_entity_mapping():
        filename = settings.ENTITY_TO_NAME_FILE
        logger.info("Loading name to entity mapping from %s ..." % filename)
        mapping = {}
        with open(filename, "r", encoding="utf8") as f:
            for line in f:
                entity_id, name = line.strip('\n').split('\t')
                if name not in mapping:
                    mapping[name] = []
                mapping[name].append(entity_id)
        logger.info("-> %d name to entity mappings loaded." % len(mapping))
        return mapping

    @staticmethod
    def read_descriptions():
        filename = settings.ENTITY_TO_DESCRIPTION_FILE
        logger.info("Loading description mapping from %s ..." % filename)
        descriptions = []
        with open(filename, "r", encoding="utf8") as f:
            for line in f:
                key, value = line.strip('\n').split('\t')
                descriptions.append((key, value))
        logger.info("-> %d description mappings loaded." % len(descriptions))
        return descriptions

    @staticmethod
    def read_description_mapping():
        filename = settings.ENTITY_TO_DESCRIPTION_FILE
        logger.info("Loading description mapping from %s ..." % filename)
        descriptions = {}
        with open(filename, "r", encoding="utf8") as f:
            for line in f:
                key, value = line.strip('\n').split('\t')
                descriptions[key] = value
        logger.info("-> %d description mappings loaded." % len(descriptions))
        return descriptions

    @staticmethod
    def read_all_entities():
        filename = settings.ENTITY_TO_NAME_FILE
        logger.info(f"Loading entities from {filename} ...")
        entities = set()
        with open(filename, "r", encoding="utf8") as f:
            for line in f:
                key, _ = line.strip('\n').split('\t')
                entities.add(key)
        logger.info(f"-> {len(entities)} entities loaded.")
        return entities

    @staticmethod
    def read_type_frequency_mapping():
        filename = settings.TYPE_FREQUENCY_FILE
        logger.info(f"Loading type frequencies from {filename} ...")
        mapping = {}
        with open(filename, "r", encoding="utf8") as f:
            for line in f:
                lst = line.strip('\n').split('\t')
                entity_id, frequency = lst[0], lst[-1]
                frequency = int(frequency)
                mapping[entity_id] = frequency
        logger.info(f"-> {len(mapping)} type frequencies loaded.")
        return mapping

    @staticmethod
    def read_accumulated_type_popularity_mapping():
        filename = settings.TYPE_POPULARITY_FILE
        logger.info(f"Loading accumulated type popularities from {filename} ...")
        mapping = {}
        with open(filename, "r", encoding="utf8") as f:
            for line in f:
                lst = line.strip('\n').split('\t')
                entity_id, popularity = lst[0], lst[-1]
                popularity = int(popularity)
                mapping[entity_id] = popularity
        logger.info(f"-> {len(mapping)} accumulated type popularities loaded.")
        return mapping

    @staticmethod
    def read_entity_popularity_mapping():
        filename = settings.ENTITY_TO_POPULARITY_FILE
        logger.info(f"Loading entity popularities from {filename} ...")
        mapping = {}
        with open(filename, "r", encoding="utf8") as f:
            for line in f:
                entity_id, popularity = line.strip('\n').split('\t')
                popularity = int(popularity)
                mapping[entity_id] = popularity
        logger.info(f"-> {len(mapping)} entity popularities loaded.")
        return mapping

    @staticmethod
    def read_prominent_types_mapping():
        filename = settings.PROMINENT_TYPE_FILE
        logger.info(f"Loading prominent types from {filename} ...")
        mapping = {}
        with open(filename, "r", encoding="utf8") as f:
            for line in f:
                lst = line.strip('\n').split('\t')
                entity_id, prominent_type = lst[0], lst[-1]
                mapping[entity_id] = prominent_type
        logger.info(f"-> {len(mapping)} prominent types loaded.")
        return mapping

    @staticmethod
    def read_type_variance_mapping():
        filename = settings.TYPE_VARIANCE_FILE
        logger.info(f"Loading type variances from {filename} ...")
        mapping = {}
        with open(filename, "r", encoding="utf8") as f:
            for line in f:
                lst = line.strip('\n').split('\t')
                entity_id, variance = lst[0], lst[-1]
                variance = float(variance)
                mapping[entity_id] = variance
        logger.info(f"-> {len(mapping)} type variances loaded.")
        return mapping

    @staticmethod
    def read_type_to_entities():
        """
        This mapping takes > 90GB of space.
        """
        filename = settings.ALL_ENTITY_TYPES
        logger.info(f"Loading type to entities mapping from {filename} ...")
        mapping = {}
        with open(filename, "r", encoding="utf8") as f:
            for line in f:
                entity_id, type_id = line.strip('\n').split('\t')
                if type_id not in mapping:
                    mapping[type_id] = set()
                mapping[type_id].add(entity_id)
        logger.info(f"-> {len(mapping)} type to entities mappings loaded.")
        return mapping

    @staticmethod
    def read_from_db(db_file: str, value_type: Optional[type] = str, separator: Optional[str] = ",") -> Database:
        db = Database(db_file, value_type, separator)
        return db

    @staticmethod
    def read_description_db() -> Database:
        filename = settings.ENTITY_TO_DESCRIPTION_DB
        logger.info(f"Loading entity ID to description database from {filename} ...")
        description_db = EntityDatabaseReader.read_from_db(filename)
        logger.info(f"-> {len(description_db)} entity ID to description mappings loaded.")
        return description_db

    @staticmethod
    def read_name_db() -> Database:
        filename = settings.ENTITY_TO_LABEL_DB
        logger.info(f"Loading entity ID to label database from {filename} ...")
        label_db = EntityDatabaseReader.read_from_db(filename)
        logger.info(f"-> {len(label_db)} entity ID to label mappings loaded.")
        return label_db

    @staticmethod
    def read_sitelink_db() -> Database:
        filename = settings.ENTITY_TO_SITELINKS_DB
        logger.info(f"Loading entity ID to number of sitelinks database from {filename} ...")
        sitelinks_db = EntityDatabaseReader.read_from_db(filename, value_type=int)
        logger.info(f"-> {len(sitelinks_db)} entity ID to number of sitelinks mappings loaded.")
        return sitelinks_db

    @staticmethod
    def read_instance_of_db() -> Database:
        filename = settings.ENTITY_TO_P31_DB
        logger.info(f"Loading entity ID to instance of database from {filename} ...")
        instance_of_db = EntityDatabaseReader.read_from_db(filename, value_type=list)
        logger.info(f"-> {len(instance_of_db)} entity ID to instance of mappings loaded.")
        return instance_of_db

    @staticmethod
    def read_subclass_of_db() -> Database:
        filename = settings.ENTITY_TO_P279_DB
        logger.info(f"Loading entity ID to subclass of database from {filename} ...")
        subclass_of_db = EntityDatabaseReader.read_from_db(filename, value_type=list)
        logger.info(f"-> {len(subclass_of_db)} entity ID to subclass of mappings loaded.")
        return subclass_of_db