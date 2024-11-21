import math
import random
import logging

from src.models.entity_database import EntityDatabase

random.seed(42)

logger = logging.getLogger("main." + __name__.split(".")[-1])


class FeatureScores:
    def __init__(self, entity_db=None):
        # Load entity database mappings if they have not already been loaded
        self.entity_db = entity_db if entity_db else EntityDatabase()

        self.entity_db.load_instance_of_mapping()
        self.entity_db.load_subclass_of_mapping()
        self.entity_db.load_entity_to_name()
        self.entity_db.load_accumulated_type_popularity()
        self.entity_db.load_type_frequency()

        self.n_entities = len(self.entity_db.entity_to_name)
        self.variances = None
        self.norm_variances = None
        self.norm_idfs = None
        self.norm_popularities = None

    def precompute_normalized_variances(self, predicate_variance_files, k=5, medium=None, minimum=None, maximum=None,
                                        plot_name=None):
        self.load_predicate_variances(predicate_variance_files)
        self.norm_variances = self.normalize_scores(self.variances.items(), k, medium, minimum, maximum, plot_name)

    def precompute_normalized_idfs(self, k=5, medium=None, minimum=None, maximum=None, plot_name=None):
        self.idf_scores = [(t, math.log(self.n_entities / self.entity_db.get_type_frequency(t)))
                      for t in self.entity_db.type_frequency.keys()]
        self.norm_idfs = self.normalize_scores(self.idf_scores, k, medium, minimum, maximum, plot_name)

    def precompute_idfs(self):
        self.idf_scores = {t: math.log(self.n_entities / self.entity_db.get_type_frequency(t))
                      for t in self.entity_db.type_frequency.keys()}

    def precompute_normalized_popularities(self):
        avg_log_pop_scores = {}
        minimum = 1000
        maximum = 0
        for e, pop in self.entity_db.accumulated_type_popularity.items():
            frequency = self.entity_db.get_type_frequency(e)
            avg_log_pop = math.log(1 + pop / frequency)  # avg popularity can be 0 therefore +1
            avg_log_pop_scores[e] = avg_log_pop
            if avg_log_pop < minimum:
                minimum = avg_log_pop
            if avg_log_pop > maximum:
                maximum = avg_log_pop
        self.norm_popularities = {e: FeatureScores.min_max_normalize(score, minimum, maximum)
                                  for e, score in avg_log_pop_scores.items()}

    def get_normalized_variance(self, type_id):
        if not self.norm_variances:
            logger.warning(f"get_normalized_variance() called without having called precompute_normalized_variances().")
        return self.norm_variances[type_id] if type_id in self.norm_variances else 0

    def get_normalized_idf(self, type_id):
        if not self.norm_idfs:
            logger.warning(f"get_normalized_idf() called without having called precompute_normalized_idfs().")
        return self.norm_idfs[type_id] if type_id in self.norm_idfs else 0

    def get_normalized_popularity(self, type_id):
        if not self.norm_popularities:
            logger.warning(f"get_normalized_popularity() called without having called precompute_normalized_popularities().")
        return self.norm_popularities[type_id] if type_id in self.norm_popularities else 0

    def get_variance(self, type_id):
        if not self.variances:
            logger.warning(f"get_variance() called without having called precompute_normalized_variances().")
        return self.variances[type_id] if type_id in self.variances else 0

    def get_idf(self, type_id):
        return self.idf_scores[type_id] if type_id in self.idf_scores else 0

    def get_average_type_popularity(self, type_id):
        return self.entity_db.get_accumulated_type_popularity(type_id)

    def load_predicate_variances(self, predicate_variance_files):
        self.variances = dict()
        for input_file in predicate_variance_files:
            with open(input_file, "r", encoding="utf8") as file:
                for line in file:
                    entity_id, variance = line.strip("\n").split("\t")
                    variance = float(variance)
                    if entity_id in self.variances and self.variances[entity_id] != variance:
                        type_name = self.entity_db.get_entity_name(entity_id)
                        logger.info(f"Type {type_name} ({entity_id}) exists already with score "
                                    f"{self.variances[entity_id]:.4f} vs. {variance:.4f}")
                    else:
                        self.variances[entity_id] = variance

    @staticmethod
    def min_max_normalize(x, min, max):
        return (x - min) / (max - min)

    @staticmethod
    def normalize(x, medium, min, max, k=5):
        """
        Normalize x to a value between 0 and 1, where a maximized normalized value
        of 1 is achieved when x=medium. The curve is less steep towards minimum and
        maximum values and smoothed around the medium value.
        k controls the steepness of the curve.
        """
        # score = (medium_variance - abs(medium_variance - curr_sum)) / medium_variance
        # 1-(1-\exp(-5*((x-10)/10)^{2}))/(1+\exp(-5))
        # Slope left and right of medium should depend on the distance of medium to the min or max respectively,
        # therefore use two functions, one for the left side and one for the right side.
        if x <= medium:
            # 1\ -\ (1 -\exp(-5 * ((x-6) / (6)) ^ {2})) / (1 +\exp(-5 * ((x-6) / (6)) ^ {2}))
            return 1 - (1 - math.e ** (-k * ((x - medium) / (medium - min)) ** 2)) / (1 + math.e ** (-k * ((x - medium) / (medium - min)) ** 2))
        else:
            # 1\ -\ (1-\exp(-5*((x-6)/(6))^{2}))/(1+\exp(-5*((x-6)/(30-6))^{2}))
            return 1 - (1 - math.e ** (-k * ((x - medium) / (medium - min)) ** 2)) / (1 + math.e ** (-k * ((x - medium) / (max - medium)) ** 2))

    @staticmethod
    def normalize_scores(scores, k=5, medium=None, minimum=None, maximum=None, plot_name=None):
        """
        Compute a normalized score based on the given scores.
        The score should be smallest / 0 for very large and very small values
        and largest for values in the middle of the spectrum.
        The score should be normalized to values between 0 and 1 to be comparable
        to values for other types"""
        if not scores:
            return scores
        scores = sorted(scores, key=lambda x: x[1])
        half_sum = sum([v[1] for v in scores]) / 2
        half_sum_idx = 0
        curr_sum = 0
        if not medium:
            for i, s in enumerate(scores):
                curr_sum += s[1]
                if curr_sum >= half_sum:
                    half_sum_idx = i - 1 if i > 0 else i
                    break
            medium = scores[half_sum_idx][1]
        else:
            for i, s in enumerate(scores):
                if s[1] >= medium:
                    half_sum_idx = i
                    break
        print(f"Maximum normalized value is {medium} at index {half_sum_idx} of {len(scores)}")

        normalized_scores = {}
        maximum = max(scores[-1][1], medium + 1) if not maximum else maximum
        minimum = min(scores[0][1], max(medium - 1, 0)) if not minimum else minimum
        print(f"Normalization maximum: {maximum}, minimum: {minimum}")
        for i, s in enumerate(scores):
            normalized_scores[s[0]] = FeatureScores.normalize(s[1], medium, min=minimum, max=maximum, k=k)
        if plot_name:
            # Plot the normalized scores as line plot
            import matplotlib.pyplot as plt
            plt.plot([v[1] for v in scores])
            plt.plot(half_sum_idx, medium, 'ro', markersize=3)
            # Add labels and title
            plt.xlabel('Types')
            plt.ylabel('Normalized score')
            # Save the plot to a PDF file
            plt.savefig(f'{plot_name}.pdf')
            print(f"half sum of all scores is at value: {medium} at index {half_sum_idx + 1} of {len(scores)}")
        return normalized_scores
