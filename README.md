# Natural Entity Types

This repository contains code to select the most natural type for Wikidata entities.
For each entity, we consider the types connected to this entity in Wikidata via an instance-of/subclass-of* path.
That is, we create a set of candidate types that consists of all types that are reachable from the entity via a chain
of relations that starts with a single instance-of relation, followed by an arbitrary number of subclass-of relations
(which may be 0).
We employ various methods such as a Gradient Boost Regressor model and a feed forward neural network to score each
candidate type and select the most natural one.

## Setup with Docker

Get the code and build the docker image:

    git clone https://github.com/ad-freiburg/natural-entity-types.git
    cd natural-entity-types
    docker build -t natural_entity_types .

Run the docker container:

    docker run -it -p 8000:8000 -v $(pwd)/data/:/data -v $(pwd)/models/:/home/models -v $(pwd)/benchmarks/:/home/benchmarks natural-entity-types

Make sure the mounted directories are writable from within the docker container, e.g. by running:

    chmod a+rw -R data/ models/ benchmarks/

Inside the docker container, get the data by running:

    make download_all

OR ALTERNATIVELY, if you want the most up-to-date data, generate it by running:

    make generate_all

This will download Wikidata mappings using the [QLever](https://qlever.cs.uni-freiburg.de/wikidata) API, generate
databases from them for quick access, and compute type properties from these Wikidata mappings which are used as
features by the models. This can take a couple of hours.

You can now train and evaluate models, or used trained models to generate natural type triples for all entities in
Wikidata as described in the next sections.


## Training and Evaluation

To train and/or evaluate a model, adjust the following command according to your needs:

    python3 scripts/evaluate.py -m <gbr|nn|gpt|oracle> --save_model <model_file> -b <benchmark_file> -train <training_file>

- The `-m` option specifies the model to be evaluated. You can choose from `gbr` (Gradient Boost Regressor),
`nn` (Feed Forward Neural Network), `gpt` (GPT-4), and `oracle`. `oracle` is a model that always predicts the ground
  truth type of the benchmark entity if it is among the candidate types. The oracle evaluation results represent the
  upper bound of what a model that relies on the candidate types can achieve.
- `<model_file>` is the path where the trained model will be saved (optional).
- `<benchmark_file>` is the path to the benchmark file on which the model will be evaluated, e.g.
`benchmarks/mini_benchmark.test.tsv`. The expected format of the benchmark is a tsv file with one line per entity, with
  the entity QID in the first column and the space-separated QIDs of the ground truth types in the second column.
- `<training_file>` is the path to the file that contains the training data. The expected format is the same as for the
  benchmark file.

Once you have evaluated a model for the first time (assuming you used the `--save_model` option), you can replace
the `--save_model` option with the `--load_model` option to load the previously saved model from the specified file
without having to train it again.

## Generate Natural Type Triples

To generate natural type triples for all entities that have an `instance of` or `subclass of` relation in Wikidata, run:

    make triples

This will generate a file `data/results/natural_types.ttl` that contains the natural type triples in TTL format.