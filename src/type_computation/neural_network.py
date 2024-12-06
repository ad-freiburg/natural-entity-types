from typing import Iterator, Tuple, Optional
import torch
import spacy
import random
import math
import logging
from functools import lru_cache

from src.evaluation.benchmark_reader import BenchmarkReader
from src.evaluation.evaluation import evaluate_batch_prediction
from src.evaluation.metrics import MetricName
from src.models.entity_database import EntityDatabase
from src.type_computation.feature_scores import FeatureScores


# Ensure reproducibility
torch.manual_seed(42)
random.seed(246)


logger = logging.getLogger("main." + __name__.split(".")[-1])


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
    def __init__(self, entity_db=None):
        # Load entity database mappings if they have not been loaded already
        self.entity_db = entity_db if entity_db else EntityDatabase()
        self.entity_db.load_instance_of_mapping()
        self.entity_db.load_subclass_of_mapping()
        self.entity_db.load_entity_to_name()
        self.entity_db.load_entity_to_description()

        self.feature_scores = FeatureScores(self.entity_db)
        self.feature_scores.precompute_normalized_popularities()
        self.feature_scores.precompute_normalized_idfs()
        self.feature_scores.precompute_normalized_variances()

        logger.info("Loading spacy model...")
        self.nlp = spacy.load("en_core_web_lg")
        self.embedding_size = 300
        self.n_features = 8 + self.embedding_size*2

        self.model = None

        self.type_embedding_cache = {}
        self.desc_embedding_cache = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def initialize_model(self, hidden_layer_sizes=512, hidden_layers=1, dropout=0.4, activation="sigmoid"):
        """
        Initialize a PyTorch model with the given parameters.

        Parameters:
        - input_size (int): Number of input features.
        - output_size (int): Number of output features (e.g., number of classes for classification).
        - hidden_layer_sizes (int): Number of neurons in each hidden layer.
        - hidden_layers (int): Number of hidden layers.
        - dropout (float): Dropout probability (0 means no dropout).
        - activation (str): Activation function ('relu', 'tanh', 'sigmoid').

        Returns:
        - model (nn.Module): A PyTorch model.
        """
        # Define a dictionary for supported activation functions
        activations = {
            "relu": torch.nn.ReLU,
            "tanh": torch.nn.Tanh,
            "sigmoid": torch.nn.Sigmoid,
            "leaky_relu": torch.nn.LeakyReLU
        }

        # Check if the provided activation is supported
        if activation not in activations:
            raise ValueError(f"Unsupported activation function: {activation}. Choose from {list(activations.keys())}")

        layers = []

        # Input layer
        input_size = self.n_features
        layers.append(torch.nn.Linear(input_size, hidden_layer_sizes))
        layers.append(activations[activation]())
        layers.append(torch.nn.Dropout(dropout))

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(torch.nn.Linear(hidden_layer_sizes, hidden_layer_sizes))
            layers.append(activations[activation]())
            layers.append(torch.nn.Dropout(dropout))

        # Output layer
        layers.append(torch.nn.Linear(hidden_layer_sizes, 1))

        # Add Sigmoid activation to ensure outputs are between 0 and 1
        layers.append(torch.nn.Sigmoid())

        # Define the model as a sequential container
        self.model = torch.nn.Sequential(*layers)

    @lru_cache(maxsize=1000000)
    def get_text_embedding(self, string):
        if not string:
            return torch.zeros(1, self.embedding_size)

        doc = self.nlp(string)
        embedding = torch.zeros(1, self.embedding_size)
        for tok in doc:
            embedding += tok.vector
        embedding = embedding / len(doc)
        return embedding

    def create_feature_vector(self, type_id, path_length, desc, desc_embedding, entity_name):
        norm_pop = self.feature_scores.get_normalized_popularity(type_id)
        norm_var = self.feature_scores.get_normalized_variance(type_id)
        norm_idf = self.feature_scores.get_normalized_idf(type_id)
        type_name = self.entity_db.get_entity_name(type_id)
        type_name_embedding = self.get_text_embedding(type_name)
        type_in_desc = type_name.lower() in desc.lower() if type_name and desc else False
        len_type_name = len(type_name) if type_name else 0
        len_desc = len(desc) if desc else 0
        type_in_label = type_name.lower() in entity_name.lower() if type_name and entity_name else False
        features = [norm_pop, norm_var, norm_idf, path_length, type_in_desc, len_type_name, len_desc, type_in_label]
        return torch.cat((torch.Tensor(features).unsqueeze(0), desc_embedding, type_name_embedding), dim=1)

    def create_dataset(self, filename: str, cols_to_shuffle: Optional[Tuple[int, int]] = None, return_entity_index=False):
        """
        Create a dataset from the given file with one row per entity.
        Returns a matrix X with one row per features of an entity - candidate
        type pair and a vector y with the labels.
        cols_to_shuffle: Tuple of two integers representing the range of
        columns to shuffle in order to evaluate the effect of a feature (range).
        The first column index is inclusive, the second one exclusive.
        """
        logger.info(f"Creating dataset from {filename}...")
        dataset = BenchmarkReader.read_benchmark(filename)
        X = []
        y = []
        entity_index = {}
        for i, (e, gt_types) in enumerate(dataset.items()):
            # Add a row for each entity - candidate type pair.
            candidate_types = self.entity_db.get_entity_types_with_path_length(e)
            desc = self.entity_db.get_entity_description(e)
            desc_embedding = self.get_text_embedding(desc)
            entity_name = self.entity_db.get_entity_name(e)
            for t, path_length in candidate_types.items():
                sample_vector = self.create_feature_vector(t, path_length, desc, desc_embedding, entity_name)
                X.append(sample_vector)
                y.append(int(t in gt_types))
                if e not in entity_index:
                    entity_index[e] = []
                entity_index[e].append((len(y) - 1, t))
            print(f"\rAdded sample {i + 1}/{len(dataset)} to dataset.", end="")
        print()
        X = torch.cat(X, dim=0)
        # If cols_to_shuffle is set, shuffle the values in the specified column
        if cols_to_shuffle is not None:
            logger.info(f"Shuffling columns {cols_to_shuffle[0]} to {cols_to_shuffle[1]} ...")
            shuffled_cols = X[torch.randperm(X.size(0)), cols_to_shuffle[0]:cols_to_shuffle[1]]
            X[:, cols_to_shuffle[0]:cols_to_shuffle[1]] = shuffled_cols
        logger.info(f"Shape of X: {X.shape}")
        y = torch.Tensor(y)
        if return_entity_index:
            return X, y, entity_index
        return X, y

    def create_dataset_from_qids(self, qids: list):
        X = []
        idx_to_ent_type_pair = []
        for i, e in enumerate(qids):
            # Add a row for each entity - candidate type pair.
            candidate_types = self.entity_db.get_entity_types_with_path_length(e)
            desc = self.entity_db.get_entity_description(e)
            desc_embedding = self.get_text_embedding(desc)
            entity_name = self.entity_db.get_entity_name(e)
            for t, path_length in candidate_types.items():
                sample_vector = self.create_feature_vector(t, path_length, desc, desc_embedding, entity_name)
                X.append(sample_vector)
                idx_to_ent_type_pair.append((e, t))
        X = torch.cat(X, dim=0)
        return X, idx_to_ent_type_pair

    def train(self,
              x_train: torch.Tensor,
              y_train: torch.Tensor,
              n_epochs: int = 100,
              batch_size: int = 64,
              learning_rate: float = 0.0001,
              optimizer: str = "adam",
              momentum: float = 0,
              patience: int = 5,
              X_val: torch.Tensor = None,
              y_val: torch.Tensor = None,
              entity_index: dict = None,
              val_benchmark: dict = None):
        """
        Train the neural network.
        """
        # Initialize variables for Early Stopping
        best_val_hit_rate = 0
        patience_counter = 0
        best_model_state = None
        if X_val is not None and y_val is not None and entity_index is not None:
            y_val = y_val.unsqueeze(-1)
            X_val, y_val = X_val.to(self.device), y_val.to(self.device)
        elif X_val is not None or y_val is not None or entity_index is not None:
            logger.warning(f"X_val, y_val and entity_index must be provided for validation.")

        # Move the model and training data to the device (CPU or GPU)
        self.model = self.model.to(self.device)
        x_train, y_train = x_train.to(self.device), y_train.to(self.device)

        self.model.train()
        loss_function = torch.nn.BCELoss()

        if optimizer == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)
        loss = 0
        for i in range(n_epochs):
            batches = training_batches(x_train, y_train, batch_size)
            for j, (X_batch, y_batch) in enumerate(batches):
                y_hat = self.model(X_batch)
                loss = loss_function(y_hat, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if X_val is not None and y_val is not None and entity_index is not None:
                self.model.eval()
                with torch.no_grad():
                    y_val_pred = self.model(X_val)
                    evaluation_results = evaluate_batch_prediction(y_val_pred, val_benchmark, entity_index, [MetricName.HIT_RATE_AT_1])
                    hit_rate = evaluation_results[MetricName.HIT_RATE_AT_1]

                # Check for improvement
                if hit_rate > best_val_hit_rate:
                    best_val_hit_rate = hit_rate
                    patience_counter = 0  # Reset patience counter
                    # Store the best model
                    best_model_state = self.model.state_dict()
                else:
                    patience_counter += 1
                    logger.debug(f"No improvement for {patience_counter} epoch(s).")

                # Early stopping condition
                if patience_counter >= patience and best_model_state is not None:
                    logger.info(f"Early stopping triggered at epoch {i + 1}")
                    self.model.load_state_dict(best_model_state)
                    break

                self.model.train()

            logger.info(f"epoch {i + 1}, loss: {float(loss)}")
        self.model.eval()

    def predict(self, entity_id):
        candidate_types = self.entity_db.get_entity_types_with_path_length(entity_id)
        candidate_types = list(candidate_types.items())
        desc = self.entity_db.get_entity_description(entity_id)
        desc_embedding = self.get_text_embedding(desc)
        X = torch.Tensor()
        entity_name = self.entity_db.get_entity_name(entity_id)
        for t, path_length in candidate_types:
            sample_vector = self.create_feature_vector(t, path_length, desc, desc_embedding, entity_name)
            X = torch.cat((X, sample_vector), dim=0)
        if X.shape[0] == 0:
            logger.debug(f"Entity does not seem to have any type.")
            return None
        X = X.to(self.device)
        prediction = self.model(X)
        sorted_indices = torch.argsort(prediction.view(-1))
        sorted_indices = sorted_indices.flip([0])
        return sorted([(prediction[i], candidate_types[i][0]) for i in sorted_indices], key=lambda x: x[0], reverse=True)

    def predict_batch(self, X):
        """
        Predict the types for a batch of entities.
        """
        return self.model(X)

    def save_model(self, model_path):
        """
        Save the model and its settings in a dictionary.
        """
        torch.save({'model': self.model}, model_path)
        logger.info(f"Saved trained model to {model_path}")

    def load_model(self, model_path):
        """
        Load the model and its settings from a dictionary.
        """
        logger.info(f"Loading model from {model_path} ...")
        model_dict = torch.load(model_path)
        self.model = model_dict['model']
