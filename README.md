# Natural Entity Types

This repository contains code to select the most natural type for Wikidata entities.
For each entity, we consider the types connected to this entity in Wikidata via an instance-of/subclass-of* path.
That is, we create a set of candidate types that consists of all types that are reachable from the entity via a chain
of relations that starts with a single instance-of relation, followed by an arbitrary number of subclass-of relations
(which may be 0).
We employ various methods such as a Gradient Boost Regressor model and a feed forward neural network to score each
candidate type and select the most natural one.

## Setup

TODO

## Evaluation

To evaluate a model, adjust and run the following command:

    python3 scripts/evaluate.py -m <gbr|nn|gpt> --save_model <model_file> -b <benchmark_file> -i data/predicate_variances.Q* -train <training_file>

- The `-m` option specifies the model to be evaluated. You can choose from `gbr` (Gradient Boost Regressor),
`nn` (Feed Forward Neural Network), and `gpt` (GPT-4).
- `<model_file>` is the path where the trained model should be saved (optional).
- `<benchmark_file>` is the path to the benchmark file on which the model will be evaluated, e.g.
`benchmarks/benchmarks/mini_benchmark.test.tsv`.
- `<training_file>` is the path to the file that contains the training data.

Once you have evaluated a model for the first time (assuming you used the `--save_model` option), you can replace
the `--save_model` option with the `--load_model` option to load the previously saved model from the specified file.