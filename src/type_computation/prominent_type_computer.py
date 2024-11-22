import math
import random
import logging

from src.models.entity_database import EntityDatabase
from src.type_computation.feature_scores import FeatureScores

random.seed(42)

logger = logging.getLogger("main." + __name__.split(".")[-1])


class ProminentTypeComputer:
    def __init__(self, output_file=None, entity_db=None):
        self.output_file = output_file

        # Load entity database mappings if they have not been loaded already
        self.entity_db = entity_db if entity_db else EntityDatabase()
        self.entity_db.load_instance_of_mapping()
        self.entity_db.load_subclass_of_mapping()
        self.entity_db.load_entity_to_name()
        self.entity_db.load_accumulated_type_popularity()
        self.entity_db.load_type_frequency()

        self.n_entities = len(self.entity_db.entity_to_name)

        self.feature_scores = FeatureScores(self.entity_db)
        self.feature_scores.precompute_normalized_variances(medium=0.8,
                                                            plot_name="normlized_variances")
        self.feature_scores.precompute_normalized_idfs(medium=7, plot_name="normalized_idf_scores")
        self.feature_scores.precompute_normalized_popularities()

    def compute_entity_score(self, entity_id, verbose=False):
        entity_name = self.entity_db.get_entity_name(entity_id)
        if not entity_name:
            if verbose:
                logger.info(f"No entity with entity ID {entity_id} found in entity to name mapping.")
            return None
        types = self.entity_db.get_entity_types(entity_id)

        type_scores = []
        not_scored = []
        for t in types:
            normalized_idf = self.feature_scores.get_normalized_idf(t)
            normalized_variance = self.feature_scores.get_normalized_variance(t)
            normalized_popularity = self.feature_scores.get_normalized_popularity(t)
            score = (3 * normalized_popularity + normalized_idf + normalized_variance) / 4
            type_name = self.entity_db.get_entity_name(t)
            type_scores.append((score, t, type_name, normalized_idf, normalized_popularity, normalized_variance))
        sorted_types = sorted(type_scores, reverse=True)
        if verbose:
            print(f"**** {entity_name} ({entity_id}) ****")
            for string in self.iter_result_string(sorted_types, not_scored):
                print(string, end="")
        return sorted_types

    def iter_result_string(self, sorted_types, not_scored=None):
        max_d = len(str(self.n_entities))
        for score, t, type_name, normalized_idf, norm_avg_pop, variance_score in sorted_types:
            variance = self.feature_scores.get_variance(t)
            frequency = self.entity_db.get_type_frequency(t)
            inverse_frequency = math.log(self.n_entities / frequency)
            popularity = self.entity_db.get_accumulated_type_popularity(t)
            avg_pop = popularity / frequency
            yield (
                f"\tType freq.: {frequency:{max_d}d} | "
                f"norm. idf: {normalized_idf:.3f} ({inverse_frequency:5.2f}) | "
                f"norm avg. pop.: {norm_avg_pop:.2f} ({avg_pop:6.2f}) | "
                f"norm. var.: {variance_score:.3f} ({variance:5.2f}) | "
                f"score: {score:6.2f}\t{type_name} ({t})\n")
        if not_scored:
            for t in not_scored:
                type_name = self.entity_db.get_entity_name(t)
                yield f"{type_name} ({t}), "
        yield "\n"

    def compute_all_entity_scores(self, n=-1):
        prominent_types = dict()
        count = 0
        for entity_id in self.entity_db.entity_to_name.keys():
            sorted_types = self.compute_entity_score(entity_id, verbose=True)
            if sorted_types:
                prominent_types[entity_id] = sorted_types
                count += 1
            if count == n:
                break
        return prominent_types

    def compute_random_entity_scores(self, n):
        prominent_types = dict()
        count = 0
        maximum = max(n * 100, 10_000)
        random_numbers = random.sample(range(0, maximum), n * 10)
        for num in random_numbers:
            entity_id = "Q" + str(num)
            sorted_types = self.compute_entity_score(entity_id, verbose=True)
            if sorted_types:
                prominent_types[entity_id] = sorted_types
                count += 1
            if count == n:
                break
        return prominent_types

    def write_entity_scores(self, prominent_types, output_file, detailed=True):
        logger.info(f"Writing prominent type mappings to {output_file} ...")
        with open(output_file, "w", encoding="utf8") as out_file:
            for entity_id, types in prominent_types.items():
                entity_name = self.entity_db.get_entity_name(entity_id)
                if detailed:
                    out_file.write(f"**** {entity_name} ({entity_id}) ****\n")
                    for string in self.iter_result_string(types):
                        out_file.write(string)
                else:
                    out_file.write(f"{entity_id} ({entity_name})\t")
                    for s, _, type_name, _, _, _ in types:
                        out_file.write(f"{type_name} ({s:.2f}); ")
                    out_file.write("\n")
        logger.info(f"Wrote {len(prominent_types)} prominent type mappings to {output_file}")
