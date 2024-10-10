import os
import openai
import time

from src.evaluation.benchmark_reader import BenchmarkReader
from src.models.entity_database import EntityDatabase


class GPT:
    def __init__(self, entity_db: EntityDatabase, model=None):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model if model else "gpt-4-1106-preview"
        self.temperature = 0.7

        # Load entity database mappings if they have not been loaded already
        self.entity_db = entity_db if entity_db else EntityDatabase()
        self.entity_db.load_instance_of_mapping()
        self.entity_db.load_subclass_of_mapping()
        self.entity_db.load_entity_to_name()

    def evaluate(self, test_file: str):
        benchmark = BenchmarkReader.read_benchmark(test_file)
        res = 0
        for e, gt_types in benchmark.items():
            # candidate_types = list(self.entity_db.get_entity_types(e))
            # for t in candidate_types:
            predicted_type_id = "42"
            if predicted_type_id in gt_types:
                res += 1
            print(
                f"Entity: {self.entity_db.get_entity_name(e)}, "
                f"prediction: {self.entity_db.get_entity_name(predicted_type_id)} vs. "
                f"{', '.join([self.entity_db.get_entity_name(gt) for gt in gt_types])}")
        accuracy = res / len(benchmark)
        print(f"Model yields correct prediction for {accuracy * 100:.1f}% of entities in the test set.")

    def predict(self, entity_id: str):
        instructions = "The user provides a Wikidata entity with name and QID, followed by a list of Wikidata types for " \
                       "the entity. You reply with the top 5 most natural types for the given entity from the " \
                       "provided type list, ordered by what the most natural type is. " \
                       "Reply with only the type QIDs separated by comma and nothing else, in particular, do not write the type names."
        entity_name = self.entity_db.get_entity_name(entity_id)
        candidate_types = self.entity_db.get_entity_types(entity_id)
        candidate_type_strings = [f"{self.entity_db.get_entity_name(t)} ({t})"
                                  for t in candidate_types]
        user_message = f"{entity_name} ({entity_id}): [{', '.join(candidate_type_strings)}]"
        trial_num = 0
        response = None
        while trial_num < 3 and not response:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": instructions},
                        {"role": "user", "content": user_message}
                    ])
            except openai.error.ServiceUnavailableError:
                print(f"Error trying to access the OpenAI API. Trying again...")
                trial_num += 1
                time.sleep(trial_num * 5)
        predicted_type_ids = None
        if response.choices:
            predicted_type_ids = response.choices[0].message.content
            print("Prediction:", predicted_type_ids)
            predicted_type_ids = [t.strip() for t in predicted_type_ids.split(",")]
            # Filter out types that have been predicted multiple times. GPT (3.5) does that sometimes and it
            # increases the precision to over 1.0
            filtered_types = []
            for i, t in enumerate(predicted_type_ids):
                if t not in predicted_type_ids[:i]:
                    filtered_types.append(t)
            print(f"Predicted type IDs: {predicted_type_ids}")
        return predicted_type_ids
