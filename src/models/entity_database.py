import logging
from typing import Set, Optional, Dict

from src.utils.entity_database_reader import EntityDatabaseReader


logger = logging.getLogger("main." + __name__.split(".")[-1])


class EntityDatabase:
    def __init__(self):
        self.instance_of_mapping = {}
        self.reverse_instance_of_mapping = {}
        self.subclass_of_mapping = {}
        self.reverse_subclass_of_mapping = {}
        self.entity_to_name = {}
        self.name_to_entity = {}
        self.entity_to_description = {}
        self.entity_descriptions = []
        self.type_frequency = {}
        self.type_variance = {}
        self.accumulated_type_popularity = {}
        self.entity_to_popularity = {}
        self.entities = set()
        self.prominent_types = {}
        self.predicate_variances = {}

    def load_entities(self):
        self.entities = EntityDatabaseReader.read_all_entities()

    def load_instance_of_mapping(self, as_dictionary=False):
        if not self.instance_of_mapping:
            if as_dictionary:
                self.instance_of_mapping = EntityDatabaseReader.read_instance_of_mapping()
            else:
                self.instance_of_mapping = EntityDatabaseReader.read_instance_of_db()

    def load_reverse_instance_of_mapping(self):
        if not self.reverse_instance_of_mapping:
            self.reverse_instance_of_mapping = EntityDatabaseReader.read_reverse_instance_of_mapping()

    def load_subclass_of_mapping(self, as_dictionary=False):
        if not self.subclass_of_mapping:
            if as_dictionary:
                self.subclass_of_mapping = EntityDatabaseReader.read_subclass_of_mapping()
            else:
                self.subclass_of_mapping = EntityDatabaseReader.read_subclass_of_db()

    def load_reverse_subclass_of_mapping(self):
        if not self.reverse_subclass_of_mapping:
            self.reverse_subclass_of_mapping = EntityDatabaseReader.read_reverse_subclass_of_mapping()

    def load_entity_to_name(self, as_dictionary=False):
        if not self.entity_to_name:
            if as_dictionary:
                self.entity_to_name = EntityDatabaseReader.read_name_mapping()
            else:
                self.entity_to_name = EntityDatabaseReader.read_name_db()

    def load_name_to_entity(self):
        if not self.name_to_entity:
            self.name_to_entity = EntityDatabaseReader.read_name_to_entity_mapping()

    def load_entity_to_description(self, as_dictionary=False):
        if not self.entity_to_description:
            if as_dictionary:
                self.entity_to_description = EntityDatabaseReader.read_description_mapping()
            else:
                self.entity_to_description = EntityDatabaseReader.read_description_db()

    def load_entity_popularity(self, as_dictionary=False):
        if not self.entity_to_popularity:
            if as_dictionary:
                self.entity_to_popularity = EntityDatabaseReader.read_entity_popularity_mapping()
            else:
                self.entity_to_popularity = EntityDatabaseReader.read_sitelink_db()

    def load_type_frequency(self):
        if not self.type_frequency:
            self.type_frequency = EntityDatabaseReader.read_type_frequency_mapping()

    def load_type_variance(self):
        if not self.type_variance:
            self.type_variance = EntityDatabaseReader.read_type_variance_mapping()

    def load_predicate_variances(self):
        if not self.predicate_variances:
            self.predicate_variances = EntityDatabaseReader.read_predicate_variances()

    def load_accumulated_type_popularity(self):
        if not self.accumulated_type_popularity:
            self.accumulated_type_popularity = EntityDatabaseReader.read_accumulated_type_popularity_mapping()

    def load_prominent_types(self):
        if not self.prominent_types:
            self.prominent_types = EntityDatabaseReader.read_prominent_types_mapping()

    def get_entity_types(self, entity_id: str) -> Set[str]:
        if entity_id not in self.instance_of_mapping:
            return set()
        all_types = set()
        # Make a copy of the set, otherwise calling pop on the type_set will delete the
        # element from the entry in the instance_of_mapping as well.
        type_set = set(self.instance_of_mapping[entity_id])
        all_types.update(type_set)
        while type_set:
            t = type_set.pop()
            if t in self.subclass_of_mapping:
                new_types = self.subclass_of_mapping[t]
                for new_t in new_types:
                    # Avoid loops by checking if one of the new types had already been encountered before
                    if new_t not in all_types:
                        type_set.add(new_t)
                all_types.update(new_types)
        return all_types

    def get_entity_types_with_path_length(self, entity_id: str) -> Dict[str, int]:
        if entity_id not in self.instance_of_mapping:
            return {}
        all_types = {}
        # Make a copy of the set, otherwise calling pop on the type_set will delete the
        # element from the entry in the instance_of_mapping as well.
        type_set = self.instance_of_mapping[entity_id]

        for t in type_set:
            all_types[t] = 1

        curr_path_length = 2
        next_type_set = set()
        while type_set:
            t = type_set.pop()
            if t in self.subclass_of_mapping:
                new_types = self.subclass_of_mapping[t]
                for new_t in new_types:
                    # Avoid loops by checking if one of the new types had already been encountered before
                    if new_t not in all_types:
                        next_type_set.add(new_t)
                for nt in new_types:
                    all_types[nt] = curr_path_length
            if len(type_set) == 0:
                type_set = next_type_set
                curr_path_length += 1
        return all_types

    def get_instance_of_types(self, entity_id: str) -> Set[str]:
        if entity_id in self.instance_of_mapping:
            return self.instance_of_mapping[entity_id]
        return set()

    def get_parent_types(self, type_id: str) -> Set[str]:
        if type_id in self.subclass_of_mapping:
            return self.subclass_of_mapping[type_id]
        return set()

    def get_all_parent_types(self, type_id: str) -> Set[str]:
        if type_id not in self.subclass_of_mapping:
            return set()
        all_parent_types = set()
        # Make a copy of the set, otherwise calling pop on the type_set will delete the
        # element from the entry in the instance_of_mapping as well.
        type_set = set(self.subclass_of_mapping[type_id])
        all_parent_types.update(type_set)
        while type_set:
            t = type_set.pop()
            if t in self.subclass_of_mapping:
                new_types = self.subclass_of_mapping[t]
                for new_t in new_types:
                    # Avoid loops by checking if one of the new types had already been encountered before
                    if new_t not in all_parent_types:
                        type_set.add(new_t)
                all_parent_types.update(new_types)
        return all_parent_types

    def get_child_types(self, type_id: str) -> Set[str]:
        if type_id in self.reverse_subclass_of_mapping:
            return self.reverse_subclass_of_mapping[type_id]
        return set()

    def get_all_child_types(self, type_id: str) -> Set[str]:
        if type_id not in self.reverse_subclass_of_mapping:
            return set()
        all_child_types = set()
        # Make a copy of the set, otherwise calling pop on the type_set will delete the
        # element from the entry in the instance_of_mapping as well.
        type_set = set(self.reverse_subclass_of_mapping[type_id])
        all_child_types.update(type_set)
        while type_set:
            t = type_set.pop()
            if t in self.reverse_subclass_of_mapping:
                new_types = self.reverse_subclass_of_mapping[t]
                for new_t in new_types:
                    # Avoid loops by checking if one of the new types had already been encountered before
                    if new_t not in all_child_types:
                        type_set.add(new_t)
                all_child_types.update(new_types)
        return all_child_types

    def get_entity_for_instance_of_type(self, type_id: str):
        if type_id in self.reverse_instance_of_mapping:
            return self.reverse_instance_of_mapping[type_id]
        return set()

    def get_entities_for_type(self, type_id: str) -> Set[str]:
        # Add all entities that are a direct instance of the given type
        entities = self.get_entity_for_instance_of_type(type_id)
        # Get all child types of the given type
        child_types = set(self.get_child_types(type_id))
        checked_types = {type_id}
        checked_types.update(child_types)
        while child_types:
            child_type = child_types.pop()
            # Add all entities that are instance of a child type
            entities.update(self.get_entity_for_instance_of_type(child_type))
            # Get all child types of the current child type
            new_child_types = self.get_child_types(child_type)
            for ct in new_child_types:
                if ct not in checked_types:
                    child_types.add(ct)
            checked_types.update(new_child_types)
        return entities

    def get_entity_name(self, entity_id: str) -> Optional[str]:
        if entity_id in self.entity_to_name:
            return self.entity_to_name[entity_id]
        return None

    def get_entity_by_name(self, entity_name: str) -> Optional[str]:
        if entity_name in self.name_to_entity:
            entities = self.name_to_entity[entity_name]
            max_popularity = 0, entities[0]
            for e in entities:
                popularity = self.get_entity_popularity(e)
                if popularity > max_popularity[0]:
                    max_popularity = popularity, e
            return max_popularity[1]
        return None

    def get_entity_popularity(self, entity_id: str) -> int:
        if entity_id in self.entity_to_popularity:
            return self.entity_to_popularity[entity_id]
        return 0

    def get_entity_description(self, entity_id: str) -> Optional[str]:
        if entity_id in self.entity_to_description:
            return self.entity_to_description[entity_id]
        return None

    def get_type_frequency(self, entity_id: str) -> int:
        if entity_id in self.type_frequency:
            return self.type_frequency[entity_id]
        return 0

    def get_type_variance(self, entity_id: str) -> int:
        if entity_id in self.type_variance:
            return self.type_variance[entity_id]
        return 0

    def get_accumulated_type_popularity(self, entity_id: str) -> int:
        if entity_id in self.accumulated_type_popularity:
            return self.accumulated_type_popularity[entity_id]
        return 0

    def get_prominent_type(self, entity_id: str) -> Optional[str]:
        if entity_id in self.prominent_types:
            return self.prominent_types[entity_id]
        return None

    def get_predicate_variance(self, entity_id: str) -> Optional[str]:
        if entity_id in self.predicate_variances:
            return self.predicate_variances[entity_id]
        return 0