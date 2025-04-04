from enum import Enum


class ModelNames(Enum):
    GPT = "gpt"
    GRADIENT_BOOST_CLASSIFIER = "gbc"
    GRADIENT_BOOST_REGRESSOR = "gbr"
    MANUAL_SCORING = "manual_scoring"
    NEURAL_NETWORK = "nn"
    ORACLE = "oracle"  # Best possible prediction given the candidate types
    PREDICTION_READER = "prediction_reader"
    DESCRIPTION_BASELINE = "description"