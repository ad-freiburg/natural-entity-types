from typing import Iterator, Tuple
import torch
import spacy
import random
import math
import logging

from src.evaluation.benchmark_reader import BenchmarkReader
from src.models.entity_database import EntityDatabase
from src.type_computation.feature_scores import FeatureScores


# Ensure reproducibility
torch.manual_seed(42)
random.seed(246)


logger = logging.getLogger("main." + __name__.split(".")[-1])


class NeuralNet(torch.nn.Module):
    def __init__(self, in_size, hidden_size, out_size, dropout=0.5):
        super(NeuralNet, self).__init__()
        self.l1 = torch.nn.Linear(in_size, hidden_size)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.sigmoid1 = torch.nn.Sigmoid()
        self.l2 = torch.nn.Linear(hidden_size, out_size)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.sigmoid2 = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.l1(x)
        out = self.dropout1(out)
        out = self.sigmoid1(out)
        out = self.l2(out)
        out = self.dropout2(out)
        out = self.sigmoid2(out)
        return out


def training_batches(x_train: torch.Tensor,
                     y_train: torch.Tensor,
                     batch_size: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Iterator over random batches of size <batch_size> from the training data
    """
    indices = list(range(x_train.shape[0]))
    random.shuffle(indices)
    n_batches = math.ceil(x_train.shape[0] / batch_size)
    for batch_no in range(n_batches):
        begin = batch_no * batch_size
        end = begin + batch_size
        batch_indices = indices[begin:end]
        x = x_train[batch_indices]
        y = y_train[batch_indices].unsqueeze(1)
        yield x, y


class NeuralTypePredictor:
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

        self.model_path = "trained_type_prediction_model.pt"
        self.checkpoint_path = "trained_type_prediction_model.best.pt"

        logger.info("Loading spacy model...")
        self.nlp = spacy.load("en_core_web_lg")
        self.embedding_size = 300

        self.model = None

    def set_model_path(self, model_path: str):
        self.model_path = model_path

    def initialize_model(self, n_features, hidden_units, dropout):
        self.model = NeuralNet(n_features, hidden_units, 1, dropout)

    def get_text_embedding(self, string):
        if not string:
            return torch.zeros(1, self.embedding_size)

        doc = self.nlp(string)
        embedding = torch.zeros(1, self.embedding_size)
        for tok in doc:
            embedding += tok.vector
        embedding = embedding / len(doc)
        return embedding

    def create_dataset(self, filename):
        logger.info("Creating training data...")
        training_benchmark = BenchmarkReader.read_benchmark(filename)
        X = torch.Tensor()
        y = []
        for e, gt_types in training_benchmark.items():
            # Add a row for each entity - candidate type pair.
            candidate_types = self.entity_db.get_entity_types_with_path_length(e)
            desc = self.entity_db.get_entity_description(e)
            desc_embedding = self.get_text_embedding(desc)
            entity_name = self.entity_db.get_entity_name(e)
            for t, path_length in candidate_types.items():
                norm_pop = self.feature_scores.get_normalized_popularity(t)
                norm_var = self.feature_scores.get_normalized_variance(t)
                norm_idf = self.feature_scores.get_normalized_idf(t)
                type_name = self.entity_db.get_entity_name(t)
                type_name_embedding = self.get_text_embedding(type_name)
                type_in_desc = type_name.lower() in desc.lower() if type_name and desc else False
                len_type_name = len(type_name) if type_name else 0
                len_desc = len(desc) if desc else 0
                type_in_label = type_name.lower() in entity_name.lower() if type_name and entity_name else False
                features = [norm_pop, norm_var, norm_idf, path_length, type_in_desc, len_type_name, len_desc, type_in_label]
                sample_vector = torch.cat((torch.Tensor(features).unsqueeze(0), desc_embedding, type_name_embedding), dim=1)
                X = torch.cat((X, sample_vector), dim=0)
                y.append(int(t in gt_types))
        logger.info(f"X shape: {X.shape}")
        return X, torch.Tensor(y)

    def train(self,
              x_train: torch.Tensor,
              y_train: torch.Tensor,
              n_epochs: int = 200,
              batch_size: int = 16,
              learning_rate: float = 0.01):
        """
        Train the neural network.
        """
        self.model.train()
        loss_function = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        for i in range(n_epochs):
            batches = training_batches(x_train, y_train, batch_size)
            loss = 0
            for j, (X_batch, y_batch) in enumerate(batches):
                y_hat = self.model(X_batch)
                loss = loss_function(y_hat, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            logger.info(f"epoch {i + 1}, loss: {float(loss)}")

    def predict(self, entity_id):
        candidate_types = self.entity_db.get_entity_types_with_path_length(entity_id)
        candidate_types = list(candidate_types.items())
        desc = self.entity_db.get_entity_description(entity_id)
        desc_embedding = self.get_text_embedding(desc)
        X = torch.Tensor()
        entity_name = self.entity_db.get_entity_name(entity_id)
        for t, path_length in candidate_types:
            norm_pop = self.feature_scores.get_normalized_popularity(t)
            norm_var = self.feature_scores.get_normalized_variance(t)
            norm_idf = self.feature_scores.get_normalized_idf(t)
            type_name = self.entity_db.get_entity_name(t)
            type_name_embedding = self.get_text_embedding(type_name)
            type_in_desc = type_name.lower() in desc.lower() if type_name and desc else False
            len_type_name = len(type_name) if type_name else 0
            len_desc = len(desc) if desc else 0
            type_in_label = type_name.lower() in entity_name.lower() if type_name and entity_name else False
            features = [norm_pop, norm_var, norm_idf, path_length, type_in_desc, len_type_name, len_desc, type_in_label]
            sample_vector = torch.cat((torch.Tensor(features).unsqueeze(0), desc_embedding, type_name_embedding), dim=1)
            X = torch.cat((X, sample_vector), dim=0)
        if X.shape[0] == 0:
            logger.info(f"Entity does not seem to have any type.")
            return None
        prediction = self.model(X)
        sorted_indices = torch.argsort(prediction.view(-1))
        sorted_indices = sorted_indices.flip([0])
        return sorted([(prediction[i], candidate_types[i][0]) for i in sorted_indices], key=lambda x: x[0], reverse=True)

    def save_model(self):
        """
        Save the model and its settings in a dictionary.
        """
        torch.save({'model': self.model}, self.model_path)
        logger.info(f"Dictionary with trained model saved to {self.model_path}")

    @staticmethod
    def load_model(model_path):
        """
        Load the model and its settings from a dictionary.
        """
        model_dict = torch.load(model_path)
        model = model_dict['model']
        return model
