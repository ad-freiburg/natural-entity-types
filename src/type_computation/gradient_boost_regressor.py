import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from src.evaluation.benchmark_reader import BenchmarkReader
from src.models.entity_database import EntityDatabase
from src.type_computation.feature_scores import FeatureScores

SEED = 42

logger = logging.getLogger("main." + __name__.split(".")[-1])


class GradientBoostRegressor:
    def __init__(self, input_files, entity_db=None):
        # Load entity database mappings if they have not been loaded already
        self.entity_db = entity_db if entity_db else EntityDatabase()
        self.entity_db.load_instance_of_mapping()
        self.entity_db.load_subclass_of_mapping()
        self.entity_db.load_entity_to_name()
        self.entity_db.load_entity_to_description()

        self.feature_scores = FeatureScores(self.entity_db)
        self.feature_scores.precompute_normalized_popularities()
        self.feature_scores.precompute_normalized_idfs(medium=7)
        self.feature_scores.precompute_normalized_variances(input_files, medium=0.8)

        self.num_estimators = 300
        self.model = GradientBoostingRegressor(loss='squared_error',
                                               learning_rate=0.1,
                                               n_estimators=self.num_estimators,
                                               max_depth=3,
                                               random_state=SEED)

    def create_feature_list(self, type_id, path_length, desc, entity_name):
        """
        Create a list of features for a given entity type.
        """
        norm_pop = self.feature_scores.get_average_type_popularity(type_id)
        norm_var = self.feature_scores.get_normalized_variance(type_id)
        norm_idf = self.feature_scores.get_normalized_idf(type_id)
        type_name = self.entity_db.get_entity_name(type_id)
        type_in_desc = type_name.lower() in desc.lower() if type_name and desc else False
        len_type_name = len(type_name) if type_name else 0
        len_desc = len(desc) if desc else 0
        type_in_label = type_name.lower() in entity_name.lower() if type_name and entity_name else False
        return [norm_pop, norm_var, norm_idf, path_length, type_in_desc, len_type_name, len_desc, type_in_label]

    def create_dataset(self, filename):
        """
        Create a dataset from the benchmark with one row per entity - candidate
        type pair. There are three columns, one for each numerical feature
        (normalized popularity, normalized variance and normalized idf).
        The label is 1 if the candidate type is a ground truth type for the
        current entity and 0 otherwise.
        TODO: Consider adding a row for each entity - type pair. Label would be
        0 if the entity does not have that type, 1 if it is the ground truth
        type and 0.5 if the entity has the type but it is not the GT type.
        """
        benchmark = BenchmarkReader.read_benchmark(filename)
        X = []
        y = []
        for e, gt_types in benchmark.items():
            # Add a row for each entity - candidate type pair.
            candidate_types = self.entity_db.get_entity_types_with_path_length(e)
            desc = self.entity_db.get_entity_description(e)
            entity_name = self.entity_db.get_entity_name(e)
            for t, path_length in candidate_types.items():
                features = self.create_feature_list(t, path_length, desc, entity_name)
                X.append(features)
                y.append(int(t in gt_types))
        return np.array(X), y

    def train(self, X, y):
        self.model.fit(X, y)
        prediction = self.model.predict(X)
        rmse = mean_squared_error(y, prediction) ** (1 / 2)
        print(f"Root mean squared error: {rmse}")
        accuracy = self.model.score(X, y)
        print(f"Accuracy: {accuracy:.3f}")

    def evaluate(self, test_file):
        benchmark = BenchmarkReader.read_benchmark(test_file)
        res = 0
        for e, gt_types in benchmark.items():
            X_test = []
            y_test = []
            # Add a row for each entity - candidate type pair.
            candidate_types = list(self.entity_db.get_entity_types_with_path_length(e).items())
            desc = self.entity_db.get_entity_description(e)
            entity_name = self.entity_db.get_entity_name(e)
            for t, path_length in candidate_types:
                features = self.create_feature_list(t, path_length, desc, entity_name)
                X_test.append(features)
                y_test.append(int(t in gt_types))
            X_test = np.array(X_test)
            y_pred = self.model.predict(X_test)
            predicted_type_id = candidate_types[np.argmax(y_pred)][0]
            if predicted_type_id in gt_types:
                res += 1
        accuracy = res / len(benchmark)
        print(f"Model yields correct prediction for {accuracy * 100:.1f}% of entities in the test set.")

    def predict(self, entity_id, provided_candidate_types=None):
        candidate_types = self.entity_db.get_entity_types_with_path_length(entity_id)

        if provided_candidate_types is not None:
            # Get the path lengths for the provided candidate types.
            new_candidate_types = {}
            for t in provided_candidate_types:
                new_candidate_types[t] = candidate_types[t] if t in candidate_types else 0
            candidate_types = new_candidate_types

        candidate_types = list(candidate_types.items())
        desc = self.entity_db.get_entity_description(entity_id)
        X = []
        entity_name = self.entity_db.get_entity_name(entity_id)
        for t, path_length in candidate_types:
            features = self.create_feature_list(t, path_length, desc, entity_name)
            X.append(features)
        if not X:
            print(f"Entity does not seem to have any type.")
            return None
        prediction = self.model.predict(X)
        sorted_indices = np.argsort(prediction)[::-1]
        return sorted([(prediction[i], candidate_types[i][0]) for i in sorted_indices], key=lambda x: x[0], reverse=True)

    def plot_feature_importance(self, test_file):
        """
        Plot the feature importance.
        Code taken from
        https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html
        """
        from sklearn.inspection import permutation_importance
        feature_names = ["Norm pop.", "Norm var.", "Norm IDF"]
        feature_importance = self.model.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + 0.5
        fig = plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.barh(pos, feature_importance[sorted_idx], align="center")
        plt.yticks(pos, np.array(feature_names)[sorted_idx])
        plt.title("Feature Importance (MDI)")

        X_test, y_test = self.create_dataset(test_file)
        result = permutation_importance(
            self.model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
        )
        sorted_idx = result.importances_mean.argsort()
        plt.subplot(1, 2, 2)
        plt.boxplot(
            result.importances[sorted_idx].T,
            vert=False,
            labels=np.array(feature_names)[sorted_idx],
        )
        plt.title("Permutation Importance (test set)")
        fig.tight_layout()
        plt.savefig("feature_importance.pdf")

    def plot_learning_curve(self, test_file):
        """
        Plot training and test deviance over the number of estimators.
        Code taken from
        https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html
        """
        import matplotlib.pyplot as plt
        X_test, y_test = self.create_dataset(test_file)
        test_score = np.zeros((self.num_estimators,), dtype=np.float64)
        for i, y_pred in enumerate(self.model.staged_predict(X_test)):
            test_score[i] = mean_squared_error(y_test, y_pred)

        fig = plt.figure(figsize=(6, 6))
        plt.subplot(1, 1, 1)
        plt.title("Deviance")
        plt.plot(
            np.arange(self.num_estimators) + 1,
            self.model.train_score_,
            "b-",
            label="Training Set Deviance",
        )
        plt.plot(
            np.arange(self.num_estimators) + 1, test_score, "r-", label="Test Set Deviance"
        )
        plt.legend(loc="upper right")
        plt.xlabel("Boosting Iterations")
        plt.ylabel("Deviance")
        fig.tight_layout()
        plt.savefig("learning_curve.pdf")

    def save_model(self, model_path):
        from pickle import dump
        with open(model_path, "wb") as file:
            dump(self.model, file, protocol=5)
        logger.info(f"Saved model to {model_path}")


    def load_model(self, model_path):
        logger.info(f"Loading model from {model_path} ...")
        from pickle import load
        with open(model_path, "rb") as file:
            self.model = load(file)
