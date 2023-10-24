"""
Input: For a single entity, for all the types it has, input
- normalized inverse frequency
- normalized predicate variance
- normalized average popularity
Output: Get a score for each type
Rank the entity's types by that score (and take the highest one)

But how to train this? I have a input X_train with input data points and y_train with labels.
If I simply use for each entity each candidate type's feature vector as input data and whether it is a GT type as output, then I'm not learning a proper ranking.
The exact same feature vector might once occur as 0 and once as 1 depending on which other candidates exist.

Ok, but in the end, if I want a score based only on these 3 feature values, then I will end up with a global ranking over all types.
And the type from a candidate subset with the highest global score wins.

ChatGPT says I should put the features of each type into one row, so I would have one row per entity.
If an entity does not have a certain type, then the features for this type should be 0.
But that means ca. 3 * 100,000 columns in the matrix, almost all of which will be 0.
But let's try and see...
"""
import numpy as np
from scipy.sparse import csr_matrix
from lightgbm import LGBMClassifier

from src.evaluation.benchmark_reader import BenchmarkReader
from src.type_computation.feature_scores import FeatureScores


class GradientBoostClassifier:
    def __init__(self, input_files):
        self.feature_scores = FeatureScores()
        self.feature_scores.precompute_normalized_popularities()
        self.feature_scores.precompute_normalized_idfs(medium=7)
        self.feature_scores.precompute_normalized_variances(input_files, medium=0.8)
        self.entity_db = self.feature_scores.entity_db

        self.type_index = [t for t in self.entity_db.type_frequency.keys()]
        self.type_to_index = {t: i for i, t in enumerate(self.type_index)}

        self.model = LGBMClassifier()

    def create_dataset(self, filename):
        rows, cols, data = [], [], []
        y = []
        num_features = 3
        num_rows = 0
        benchmark = BenchmarkReader.read_benchmark(filename)
        for e, gt_types in benchmark.items():
            # Add a row for each entity - ground truth type pair.
            for gt_type in gt_types:
                # Add the ground truth label
                y.append(self.type_to_index[gt_type])
                # Add the data row
                candidate_types = self.entity_db.get_entity_types(e)
                for t in candidate_types:
                    norm_pop = self.feature_scores.get_normalized_popularity(t)
                    norm_var = self.feature_scores.get_normalized_variance(t)
                    norm_idf = self.feature_scores.get_normalized_idf(t)
                    features = [norm_pop, norm_var, norm_idf]
                    type_idx = self.type_to_index[t]
                    for j in range(num_features):
                        rows.append(num_rows)
                        cols.append(type_idx * 3 + j)
                        data.append(features[j])
                num_rows += 1
        num_cols = len(self.type_index) * 3
        X = csr_matrix((data, (rows, cols)), shape=(num_rows, num_cols))
        return X, y

    def train(self, X, y):
        self.model.fit(X, y)
        accuracy = self.model.score(X, y)
        print(f"Accuracy: {accuracy:.3f}")
        prediction = self.model.predict(X)
        print(prediction)
        print("\n".join([self.entity_db.get_entity_name(self.type_index[idx]) for idx in prediction]))

    def evaluate(self, filename):
        X_test, y_test = self.create_dataset(filename)
        y_pred = self.model.predict(X_test)
        pred_type_ids = [self.type_index[i] for i in y_pred]
        res = 0
        for i in range(len(pred_type_ids)):
            if pred_type_ids[i] == y_pred[i]:
                res += 1
        accuracy = res / len(y_test)
        # TODO: this is not entirely correct. Model can't receive 100% because an entity can have multiple true entities
        print(f"Model predicted the correct type for {accuracy*100:.1f}% of entities.")

    def predict(self, entity_id):
        num_features = 3
        candidate_types = self.entity_db.get_entity_types(entity_id)
        x = np.zeros((1, len(self.type_index) * 3))
        for t in candidate_types:
            norm_pop = self.feature_scores.get_normalized_popularity(t)
            norm_var = self.feature_scores.get_normalized_variance(t)
            norm_idf = self.feature_scores.get_normalized_idf(t)
            features = [norm_pop, norm_var, norm_idf]
            type_idx = self.type_to_index[t]
            for j in range(num_features):
                x[:, type_idx * 3 + j] = features[j]
        print(f"type indices: {[self.type_to_index[t] for t in candidate_types]}")
        print(f"Input vector: {x}")
        prediction = self.model.predict(x)[0]
        print(prediction)
        return [self.type_index[prediction]]

