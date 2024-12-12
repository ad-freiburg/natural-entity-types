import json

from src import settings
from src.type_computation.neural_network import NeuralTypePredictor


with open(settings.TMP_FORKSERVER_CONFIG_FILE, "r", encoding="utf8") as file:
    config = json.load(file)

if "no_loading" in config and config["no_loading"]:
    nn = None
else:
    nn = NeuralTypePredictor()
    nn.load_model(config["model_path"])
