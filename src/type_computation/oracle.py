from src.models.entity_database import EntityDatabase


class Oracle:
    def __init__(self, benchmark, entity_db=None):
        # Load entity database mappings if they have not been loaded already
        self.entity_db = entity_db if entity_db else EntityDatabase()
        self.entity_db.load_instance_of_mapping()
        self.entity_db.load_subclass_of_mapping()

        self.benchmark = None

    def predict(self, entity_id):
        """
        Predict the ground truth types for an entity if they are in the candidate types.
        The result is the best possible prediction when considering only candidate types, that is, types connected
        to the entity via an instance-of/subclass-of* path (a single instance-of relation followed by an arbitrary
        number of subclass-of relations) in the downloaded Wikidata dump.
        """
        if not self.benchmark:
            raise ValueError("No benchmark set. Use set_benchmark() to set the benchmark.")

        candidate_types = self.entity_db.get_entity_types(entity_id)
        ground_truth_types = self.benchmark[entity_id] if entity_id in self.benchmark else []
        prediction = [t for t in ground_truth_types if t in candidate_types]
        return prediction

    def set_benchmark(self, benchmark):
        self.benchmark = benchmark