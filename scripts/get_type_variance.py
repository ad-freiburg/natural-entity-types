import argparse
import sys
import numpy as np
import random
from scipy.sparse import csr_array

sys.path.append(".")

import src.utils.log as log
from src.models.entity_database import EntityDatabase
from src import settings


random.seed(42)
MAX_ENTITIES = 200_000


def main(args):
    entity_db = EntityDatabase()
    entity_db.load_reverse_instance_of_mapping()
    entity_db.load_instance_of_mapping()
    if args.debug:
        entity_db.load_entity_to_name()
    if args.subclass_of:
        entity_db.load_subclass_of_mapping()
        entity_db.load_reverse_subclass_of_mapping()
        entity_db.load_type_frequency()
        type_index = [type_id for type_id in set(entity_db.type_frequency.keys())]
    else:
        type_index = [type_id for type_id in set(entity_db.reverse_instance_of_mapping.keys())]
    logger.info(f"Checking {len(type_index)} types.")
    type_to_index = {type_id: i for i, type_id in enumerate(type_index)}
    n_types = len(type_index)

    with open(args.output_file, "w", encoding="utf8") as out_file:
        for i, type_id in enumerate(type_index):
            if args.subclass_of:
                entities = entity_db.get_entities_for_type(type_id)
            else:
                entities = entity_db.reverse_instance_of_mapping[type_id]
            if len(entities) >= MAX_ENTITIES:
                # When there are too many types for an entity, we run out of memory for the distance computation
                # So only use a random sample of entities
                entities = random.sample(entities, MAX_ENTITIES)
            type_name = entity_db.get_entity_name(type_id)
            logger.info(f"Building matrix for type {type_name} ({type_id})")
            # Build a sparse matrix with instance-of types as rows and entities as columns
            # An entry at (i, j) means that entity j is instance of type i
            rows = []
            cols = []
            data = []
            n_type_entities = len(entities)
            for j, entity in enumerate(entities):
                if args.subclass_of:
                    types = entity_db.get_entity_types(entity)
                else:
                    types = entity_db.get_instance_of_types(entity)
                for t in types:
                    if t not in type_to_index:
                        continue
                    rows.append(type_to_index[t])  # Append the type index
                    cols.append(j)  # Append the entity index
                    data.append(1)
            type_entity_matrix = csr_array((data, (rows, cols)), shape=(n_types, n_type_entities))

            # Remove all empty rows in the matrix
            non_empty_rows = type_entity_matrix.getnnz(axis=1) > 0
            type_entity_matrix = type_entity_matrix[non_empty_rows, :]
            filtered_type_index = [type_id for (type_id, non_empty) in zip(type_index, non_empty_rows) if non_empty]

            if args.debug:
                print(f"Matrix shape: {type_entity_matrix.shape}")

            # Compute the average column vector from the matrix
            avg_vector = type_entity_matrix.sum(axis=1) / type_entity_matrix.shape[1]

            if args.debug:
                for j, avg in enumerate(avg_vector):
                    typ = filtered_type_index[j]
                    typ_name = entity_db.get_entity_name(typ)
                    print(f"{avg_vector[j]:.4f}:\t{typ_name}")
                    if j == 10:
                        break
                print("\n", end="")

            # Compute euclidian distance matrix = Frobenius norm of the difference
            """
            type_entity_matrix_coo = type_entity_matrix.to_coo()
            distance_vector = np.zeros((1, type_entity_matrix.shape[1]))
            for row, col in zip(type_entity_matrix_coo.row, type_entity_matrix_coo.col):
            """

            distance_matrix = type_entity_matrix - csr_array(avg_vector).toarray().T
            distance_vector = np.sqrt(np.sum(np.square(distance_matrix), axis=0))
            type_variance = distance_vector.var()
            if args.debug:
                print(f"Variance of distances to average for type {type_name}: {type_variance}")
                print()
            out_file.write(f"{type_id}\t{type_variance:.4f}\n")

    """
    entity_db = EntityDatabase()
    entity_db.load_instance_of_mapping()
    entity_db.load_reverse_instance_of_mapping()
    entity_db.load_subclass_of_mapping()
    entity_db.load_reverse_subclass_of_mapping()
    entity_db.load_entity_to_name()

    n_entities = len(entity_db.instance_of_mapping)

    for i, type_id in enumerate(entity_db.reverse_instance_of_mapping.keys()):
        if i == 0:
            continue
        type_name = entity_db.get_entity_name(type_id)
        print(f"{type_name} ({type_id})")
        parent_types = entity_db.get_parent_types(type_id)
        while parent_types:
            new_parent_types = set()
            for t in parent_types:
                new_parent_types.update(entity_db.get_parent_types(t))
            while parent_types:
                parent_type = parent_types.pop()
                parent_type_name = entity_db.get_entity_name(parent_type)
                print(f"\t{parent_type_name}({parent_type}): ", end="")
                entities = entity_db.get_entities_for_type(parent_type)
                print(f" {len(entities)} total entities")
                if len(entities) > 0.1 * n_entities:
                    # Don't consider types which cover more than 10% of all entities.
                    print(f"\tSkipping {parent_type_name}. Parent types: {parent_types}")
                    continue
                all_entity_types = Counter()
                for entity in entities:
                    types = entity_db.get_entity_types(entity)
                    for t in types:
                        all_entity_types[t] += 1
                print("\t\t", end="")
                for t, frequency in all_entity_types.most_common(100):
                    t_name = entity_db.get_entity_name(t)
                    print(f"\"{t_name}\" ({t}): {frequency}, ", end="")
                print()
                print(f"\t\tTotal number of types: {len(all_entity_types)}")
            # Move up in the type hierarchy: Parents of previous parents are new parent types
            parent_types = new_parent_types
        if i > 10:
            break
        print()
        """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__)

    parser.add_argument("-o", "--output_file", type=str, default=settings.TYPE_VARIANCE_FILE,
                        help="File to which to write the type frequencies to.")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Print debug information.")
    parser.add_argument("-s", "--subclass-of", action="store_true",
                        help="Use instance-of/subclass-of* paths for entity types, not just an instance-of path")

    logger = log.setup_logger()

    main(parser.parse_args())
