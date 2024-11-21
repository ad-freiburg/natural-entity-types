import argparse
import sys
import numpy as np
import subprocess
import os
import random
from collections import defaultdict
from scipy.sparse import csr_array

sys.path.append(".")

import src.utils.log as log
from src.models.entity_database import EntityDatabase
from src import settings


LIMIT = 100_000_000
MAX_ENTITIES = 500_000


def plot_variances(variances, type_qid):
    # Plot the variances
    import matplotlib.pyplot as plt

    # Create a line plot
    plt.plot([v[1] for v in variances])

    half_variance = sum([v[1] for v in variances]) / 2
    medium_variance = sum([v[1] for v in variances]) / len(variances)
    half_variance_index = 0
    medium_variance_index = 0
    increases = [variances[i][1] - variances[i-1][1] for i in range(1, len(variances))]
    half_increase = sum(increases) / 2
    medium_increase = (max(increases) - min(increases)) / 2
    print(f"max increase: {max(increases)}, min increase: {min(increases)}, medium_increase: {medium_increase}")
    medium_increase_index = 0
    half_increase_index = 0
    sum_variances = 0
    sum_increases = 0
    for i in range(len(variances)):
        sum_variances += variances[i][1]
        sum_increases += increases[i-1] if i > 0 else 0
        if sum_variances >= half_variance and half_variance_index == 0:
            half_variance_index = i - 1 if i > 0 else i
        if variances[i][1] >= medium_variance and medium_variance_index == 0:
            medium_variance_index = i - 1 if i > 0 else i
        if i > 0 and variances[i][1] - variances[i-1][1] >= medium_increase and medium_increase_index == 0:
            medium_increase_index = i - 2 if i > 1 else i - 1
        if sum_increases >= half_increase and half_increase_index == 0:
            half_increase_index = i - 1
    plt.plot(half_variance_index, variances[half_variance_index][1], 'ro', markersize=3)
    plt.plot(medium_variance_index, medium_variance, 'go', markersize=3)
    plt.plot(medium_increase_index, variances[medium_increase_index][1], 'bo', markersize=3)
    plt.plot(half_increase_index, variances[half_increase_index][1], 'yo', markersize=3)
    print(f"half variance: {half_variance} at index {half_variance_index + 1} of {len(variances)}")
    print(f"medium variance: {medium_variance} at index {medium_variance_index + 1} of {len(variances)}")
    print(f"medium increase: {medium_increase} at index {medium_increase_index + 1} of {len(variances)}")
    print(f"half increase: {half_increase} at index {half_increase_index + 1} of {len(variances)}")
    # Add labels and title
    plt.xlabel('Types')
    plt.ylabel('Predicate variance')
    plt.title(f'Variance for subtypes of {type_qid}')

    # Save the plot to a PDF file
    plt.savefig(f'variances_plot.{type_qid}.pdf')


class VarianceScoreComputer:
    def __init__(self, base_type, debug, entity_db=None, write=True):
        self.base_type = base_type
        self.debug = debug
        self.write = write

        if entity_db:
            self.entity_db = entity_db
        else:
            self.entity_db = EntityDatabase()
            self.entity_db.load_reverse_subclass_of_mapping()
            self.entity_db.load_reverse_instance_of_mapping()
            self.entity_db.load_subclass_of_mapping()

            if self.debug:
                self.entity_db.load_entity_to_name()

        self.predicate_file_name = settings.PREDICATES_DIRECTORY + f"predicates.{self.base_type}.tsv"

        # Create directory if it does not exist
        if not os.path.exists(settings.PREDICATES_DIRECTORY):
            os.makedirs(settings.PREDICATES_DIRECTORY)

        # Download the predicate file if it does not exist yet
        self.success = True
        if not os.path.isfile(self.predicate_file_name):
            self.success = self.download_predicate_mapping()
            if not self.success:
                return
        else:
            logger.info(f"Using existing predicate file at {self.predicate_file_name}")

        self.predicate_index = []
        self.predicate_to_index = {}
        self.entity_index = []
        self.entity_to_index = {}

        self.build_indices()

        self.predicate_entity_matrix = None
        self.build_predicate_entity_matrix()

        self.timings = defaultdict(float)

    def download_predicate_mapping(self):
        """
        Download the mapping from entity (of type base_type) to all its
        predicates. Process the file by replacing URIs by entity / property IDs.
        """
        # Download the TSV file from QLever
        logger.info(f"Downloading entities and their predicates for type {self.base_type} to file "
                    f"{self.predicate_file_name} ...")
        tmp_file = settings.PREDICATES_DIRECTORY + "tmp_file.tsv"
        with open(tmp_file, 'w', encoding="utf8") as out_file:
            # Properties appear twice once with direct (.../prop/direct/P31) and once without (.../prop/P31).
            # I don't need both, so I use FILTER to filter them out.
            subprocess.run(['curl', '-s', 'https://qlever.cs.uni-freiburg.de/api/wikidata',
                            '-H', 'Accept: text/tab-separated-values',
                            '-H', 'Content-type: application/sparql-query',
                            '--data',
                            'PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> '
                            'PREFIX wdt: <http://www.wikidata.org/prop/direct/> '
                            'PREFIX wd: <http://www.wikidata.org/entity/> '
                            'SELECT DISTINCT ?item ?predicate WHERE { '
                            '?item wdt:P31/wdt:P279* wd:' + self.base_type + ' . '
                            '?item ?predicate [] . '
                            'FILTER(STRSTARTS(STR(?predicate), "http://www.wikidata.org/prop/P")) . '
                            'FILTER(!STRSTARTS(STR(?predicate), "http://www.wikidata.org/prop/P31")) . }'
                            ],
                           stdout=out_file)

        try:
            # Simplify the format of the file by only keeping the entity/property IDs, not the whole URI
            with open(self.predicate_file_name, "w", encoding="utf8") as out_file:
                with open(tmp_file, "r", encoding="utf8") as file:
                    for i, line in enumerate(file):
                        # Skip first line with column headers
                        if i == 0:
                            continue
                        lst = line.strip("\n").split("\t")
                        # Skip predicates like label, description, synonyms
                        if not lst[1].startswith("<http://www.wikidata.org/"):
                            continue
                        # Only keep the last element of the URI and cut off the ">" in the end
                        for j in range(len(lst)):
                            lst[j] = lst[j][lst[j].rfind("/") + 1:-1]
                        out_file.write(f"{lst[0]}\t{lst[1]}\n")

            # Remove duplicate lines. These occur because the same predicate can occur with a "direct"
            # element in its path and without it.
            with open(tmp_file, "w", encoding="utf8") as out_file:
                sort_process = subprocess.run(['sort', self.predicate_file_name], check=True,
                                              capture_output=True)
                subprocess.run(["uniq"], input=sort_process.stdout, stdout=out_file)
            subprocess.run(['mv', tmp_file, self.predicate_file_name])
            return True
        except IndexError:
            # This happens when QLever returns an error, typically because it ran out of memory
            logger.error(f"Error while processing the predicate file. QLever response was:")
            with open(tmp_file, "r", encoding="utf8") as file:
                for i, line in enumerate(file):
                    if i < 5:
                        print(line.strip())
                    else:
                        break
            # Remove the empty predicate file
            os.remove(self.predicate_file_name)
            logger.info(f"Predicate file removed.")
            return False

    def build_indices(self):
        """
        Build entity and predicate indices (from ID to index and the other way
        around) from the predicate file.
        """
        logger.info(f"Building predicate and entity indices for type {self.base_type}")
        with open(self.predicate_file_name, "r", encoding="utf8") as file:
            for i, line in enumerate(file):
                if i == 0:
                    continue
                entity_id, predicate_id = line.strip("\n").split("\t")
                if entity_id not in self.entity_to_index:
                    self.entity_index.append(entity_id)
                    self.entity_to_index[entity_id] = len(self.entity_index) - 1
                if predicate_id not in self.predicate_to_index:
                    self.predicate_index.append(predicate_id)
                    self.predicate_to_index[predicate_id] = len(self.predicate_index) - 1
        logger.info(f"Predicate file contains {len(self.predicate_index)} distinct predicates")

    def build_predicate_entity_matrix(self):
        logger.info(f"Building predicate-entity matrix from predicate file ...")
        rows = []
        cols = []
        with open(self.predicate_file_name, "r", encoding="utf8") as file:
            for i, line in enumerate(file):
                if i == 0:
                    continue
                entity_id, predicate_id = line.strip("\n").split("\t")
                entity_idx = self.entity_to_index[entity_id]
                predicate_idx = self.predicate_to_index[predicate_id]
                cols.append(entity_idx)
                rows.append(predicate_idx)
        self.predicate_entity_matrix = csr_array(([1] * len(rows), (rows, cols)),
                                                 shape=(len(self.predicate_index), len(self.entity_index)),
                                                 dtype='uint8')
        logger.info(f"Done. Predicate-entity matrix shape: {self.predicate_entity_matrix.shape}")

    def compute_variance_for_type(self, type_id: str) -> float:
        col_indices = []
        type_entities = self.entity_db.get_entities_for_type(type_id)
        # It can happen, that an entity is not in the entity_to_index mapping if the P31/P279 mappings
        # are from different Wikidata versions than the predicate mapping. Therefore, this check is needed.
        type_entities = [e for e in type_entities if e in self.entity_to_index]
        if not type_entities:
            # No entity has the given type (it is still considered a type, because it is a subclass of
            # something). Variance here does not matter, since anyhow no entity can be assigned to this type.
            return 0
        if len(type_entities) >= MAX_ENTITIES:
            # When there are too many types for an entity, we run out of memory for the distance computation
            # So only use a random sample of entities
            type_entities = random.sample(type_entities, MAX_ENTITIES)
        # Get column indices of entities that are not of type type_id
        for entity_id in type_entities:
            col_indices.append(self.entity_to_index[entity_id])
        # Create a new sparse matrix for the type with only those entities that are of type type_id
        type_predicate_entity_matrix = self.predicate_entity_matrix[:, col_indices]
        # Get row indices of predicates that no entity of type type_id has
        # Sparse matrix should contain only those predicates that at least one of the entities has
        # This does not change the distances though, so it does not change the variance
        non_empty_rows = type_predicate_entity_matrix.getnnz(axis=1) > 0
        type_predicate_entity_matrix = type_predicate_entity_matrix[non_empty_rows, :]
        if len(type_predicate_entity_matrix.shape) == 2 and type_predicate_entity_matrix.shape[1] != 0:
            child_avg_vector = type_predicate_entity_matrix.sum(axis=1) / type_predicate_entity_matrix.shape[1]
            distance_matrix = type_predicate_entity_matrix - csr_array(child_avg_vector).toarray().T
            distance_vector = np.sqrt(np.sum(np.square(distance_matrix), axis=0))
            predicate_variance = distance_vector.var()
            return predicate_variance
        return 0

    def compute_variances(self):
        logger.info(f"Computing predicate variances for type {self.base_type} and its child and parent types ...")

        # Get child types and parent types of the base type
        all_relevant_types = {self.base_type}
        all_relevant_types.update(self.entity_db.get_all_child_types(self.base_type))
        # all_relevant_types.update(self.entity_db.get_all_parent_types(self.base_type))
        variances = []
        num_types = len(all_relevant_types)
        for i, type_id in enumerate(all_relevant_types):
            print(f"\rComputing variance for type {i + 1} of {num_types} ({type_id})       ", end="")
            variances.append((type_id, self.compute_variance_for_type(type_id)))
        print()
        logger.info("Done.")

        # Write the variances to the output file
        variances = sorted(variances, key=lambda x: x[1])
        if self.debug:
            for type_id, variance in variances:
                if self.debug:
                    type_name = self.entity_db.get_entity_name(type_id)
                    print(f"{type_name} ({type_id}):\t{variance:.4f}")
        if self.write:
            # Create directory if it does not exist
            if not os.path.exists(settings.PREDICATE_VARIANCES_DIRECTORY):
                os.makedirs(settings.PREDICATE_VARIANCES_DIRECTORY)
            variance_file_name = settings.PREDICATE_VARIANCES_DIRECTORY + f"predicate_variances.{self.base_type}.tsv"
            logger.info(f"Writing variances to {variance_file_name} ...")
            with open(variance_file_name, "w", encoding="utf8") as out_file:
                for type_id, variance in variances:
                    out_file.write(f"{type_id}\t{variance:.4f}\n")
        return variances

    def compute_variance_scores(self, variances_with_info):
        """
        Compute a score based on the variance.
        The score should be smallest / 0 for very large and very small values
        and largest for values in the middle of the spectrum.
        The score should be normalized to values between 0 and 1 to be comparable
        to values for other types
        """
        variances = [v[1] for v in variances_with_info]
        variance_sum = sum(variances)
        medium_variance = variance_sum / 2
        scores = []
        curr_sum = 0
        for v in variances_with_info:
            curr_sum += v[1]
            score = (medium_variance - abs(medium_variance - curr_sum)) / medium_variance
            scores.append((v[0], score))
        scores = sorted(scores, key=lambda x: x[1])

        if self.debug:
            for type_id, score in scores:
                if self.debug:
                    type_name = self.entity_db.get_entity_name(type_id)
                    print(f"{type_name} ({type_id}):\t{score:.4f}")
        if self.write:
            variance_score_file_name = settings.DATA_DIRECTORY + f"predicate_variance_scores.{self.base_type}.tsv"
            with open(variance_score_file_name, "w", encoding="utf8") as out_file:
                for type_id, score in scores:
                    out_file.write(f"{type_id}\t{score:.4f}\n")
        return scores


def main(args):
    error_qids = []
    no_predicates_qids = []
    success_qids = []
    if "all" in args.type_qid:
        entity_db = EntityDatabase()
        entity_db.load_reverse_subclass_of_mapping()
        entity_db.load_reverse_instance_of_mapping()
        entity_db.load_subclass_of_mapping()
        second_order_types = entity_db.get_child_types("Q35120")
        third_order_types = set()
        for c in second_order_types:
            third_order_types.update(entity_db.get_child_types(c))

        variance_file_name = settings.DATA_DIRECTORY + f"predicate_variances.all.tsv"
        score_file_name = settings.DATA_DIRECTORY + f"predicate_variance_scores.all.tsv"
        variance_out_file = open(variance_file_name, "w", encoding="utf8")
        score_out_file = open(score_file_name, "w", encoding="utf8")
        logger.info(f"Computing variances for {len(third_order_types)} types ...")
        # logger.info(f"Selected types: {', '.join(third_order_types)}")
        for i, base_type in enumerate(third_order_types):
            variance_computer = VarianceScoreComputer(base_type, args.debug, entity_db=entity_db, write=not args.no_write)
            if not variance_computer.success:
                error_qids.append(base_type)
                continue
            if len(variance_computer.predicate_index) == 0:
                logger.info(f"Skipping type {base_type} with no entity predicates.")
                no_predicates_qids.append(base_type)
                continue
            variance_computer.compute_variances()
            success_qids.append(base_type)
            logger.info(f"DONE computing variance for type {i+1} out of {len(third_order_types)}")

            # for type_id, variance in variances:
            #     variance_out_file.write(f"{type_id}\t{variance:.4f}\n")
            # scores = variance_computer.compute_variance_scores(variances)
            # for type_id, score in scores:
            #     score_out_file.write(f"{type_id}\t{score:.4f}\n")
    else:
        entity_db = EntityDatabase()
        entity_db.load_reverse_subclass_of_mapping()
        entity_db.load_reverse_instance_of_mapping()
        entity_db.load_subclass_of_mapping()

        if args.debug:
            entity_db.load_entity_to_name()

        for qid in args.type_qid:
            variance_computer = VarianceScoreComputer(qid, args.debug, entity_db=entity_db, write=not args.no_write)
            if not variance_computer.success:
                error_qids.append(qid)
                continue
            if len(variance_computer.predicate_index) == 0:
                logger.info(f"Skipping type {qid} with no entity predicates.")
                no_predicates_qids.append(qid)
                continue
            variances = variance_computer.compute_variances()
            success_qids.append(qid)

            if args.generate_plot:
                variance_computer.compute_variance_scores(variances)
                plot_variances(variances, args.type_qid)

    logger.info(f"Successfully computed variances for {len(success_qids)} QIDs: {', '.join(success_qids)}")
    if no_predicates_qids:
        logger.info(f"Skipped {len(no_predicates_qids)} QIDs with no entity predicates: {', '.join(no_predicates_qids)}")
    if error_qids:
        logger.info(f"Could not compute variances for {len(error_qids)} QIDs: {', '.join(error_qids)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__)

    parser.add_argument("-qid", "--type_qid", type=str, required=True, nargs="+",
                        help="QID of the type for which to fetch all entities with their predicates.")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Print debug information.")
    parser.add_argument("-p", "--generate_plot", action="store_true",
                        help="Generate a plot of the predicate variances.")
    parser.add_argument("-nw", "--no_write", action="store_true",
                        help="Don't write the computed variances to file.")

    logger = log.setup_logger()

    main(parser.parse_args())
