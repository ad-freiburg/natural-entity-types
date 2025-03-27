import difflib

from src.models.entity_database import EntityDatabase


def get_overlap(entity_description, type_name):
    s = difflib.SequenceMatcher(None, entity_description, type_name)
    pos_a, pos_b, size = s.find_longest_match(0, len(entity_description), 0, len(type_name))
    completely_included_bonus = 1000 if type_name in entity_description else 0
    return len(entity_description[pos_a:pos_a+size]) + completely_included_bonus


class DescriptionBaseline:
    def __init__(self, entity_db: EntityDatabase):
        # Load entity database mappings if they have not been loaded already
        self.entity_db = entity_db if entity_db else EntityDatabase()
        self.entity_db.load_instance_of_mapping()
        self.entity_db.load_subclass_of_mapping()
        self.entity_db.load_entity_to_name()
        self.entity_db.load_entity_to_description()

    def predict(self, entity_id: str):
        entity_desc = self.entity_db.get_entity_description(entity_id)
        entity_desc = "" if entity_desc is None else entity_desc
        candidate_types = self.entity_db.get_entity_types_with_path_length(entity_id)
        predictions = []
        for type_id, path_length in candidate_types.items():
            type_name = self.entity_db.get_entity_name(type_id)
            type_name = "" if type_name is None else type_name
            overlap = get_overlap(entity_desc, type_name)
            predictions.append((type_id, overlap, path_length))
        predictions.sort(key=lambda pair: (pair[1], -pair[2]), reverse=True)
        predictions = [type_id for type_id, overlap, path_length in predictions]
        return predictions
