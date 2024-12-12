import re


class PredictionReader:
    def __init__(self, prediction_file):
        self.prediction_file = prediction_file
        self.predictions = self.read_predictions()
        self.prediction_index = 0

    def read_predictions(self):
        predictions = []
        with open(self.prediction_file, "r") as in_file:
            for line in in_file:
                types = re.findall(r"Q[1-9][0-9]+", line)
                predictions.append(types)
        return predictions

    def predict(self, entity):
        if self.prediction_index >= len(self.predictions):
            return []
        prediction = self.predictions[self.prediction_index]
        self.prediction_index += 1
        return prediction