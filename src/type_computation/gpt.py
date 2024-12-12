import os
import openai
import time
import re

from src.models.entity_database import EntityDatabase


SYSTEM_PROMPT = """Objective:
From a given list of Wikidata types with their label and QID, choose the most natural, broad, everyday-language type for a Wikidata item based on the item's label and description.

Rules:
- Pick the broader category if multiple levels of specificity exist (e.g. choose "Disease" over "Infectious Disease", "RNA" over "Non-coding RNA", "Star" over "Variable Star", etc.).
- If a more specific category is a commonly recognized everyday category, choose the more specific one (e.g. choose "Lake" over "Body of Water", "Village" over "Human Settlement", etc.).
- Again, avoid unnecessary specificity (e.g. choose "Surname" over "Japanese Surname", "Monument" over "Heritage Monument", etc.).
- Generally speaking, good types are short and intuitive, while bad types are long and overly specific.
- Your choice must be one of the pre-selected types.
- Return the QIDs of the top-5 most natural types, separated by comma and nothing else.

**Resist the temptation of too much specificity**  Do NOT select an overly specific type from the pre-selection simply because it appears in the item's description.
For instance, if an item has the description 'Common Japanese Surname', choose 'Surname' over the more specific 'Japanese Surname'.
Generally speaking, good types are short and intuitive, while bad types are long and overly specific.

Examples:
- Berlin → City
- Albert Einstein → Person
- T-Shirt → Clothing
- Germany → Country
- Carbon Dioxide → Chemical Compound
- Breaking Bad → Television Series
- Jazz → Musical Genre
- Sagrada Família → Church
- Green Tea → Drink
- FC Bayern Munich → Sports Club (Football Club would be too specific)

Important: A type as long and specific as e.g. "civil parish in Ireland" will **almost never** be a good choice (just "civil parish" would be much better).
Remember, types should be short and intuitive. They should be maximally broad while still being a commonly recognized, distinct category."""


def get_qids(string):
    qids = re.findall(r"Q[1-9][0-9]+", string)
    # Filter out types that have been predicted multiple times. GPT (3.5) does that sometimes and it
    # increases the precision to over 1.0
    filtered_qids = []
    for t in qids:
        if t not in filtered_qids:
            filtered_qids.append(t)
    return filtered_qids

class GPT:
    def __init__(self, entity_db: EntityDatabase, model=None):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model if model else "gpt-4o"
        # self.temperature = 0.7

        # Load entity database mappings if they have not been loaded already
        self.entity_db = entity_db if entity_db else EntityDatabase()
        self.entity_db.load_instance_of_mapping()
        self.entity_db.load_subclass_of_mapping()
        self.entity_db.load_entity_to_name()
        self.entity_db.load_entity_to_description()

    def predict(self, entity_id: str):
        entity_name = self.entity_db.get_entity_name(entity_id)
        candidate_types = self.entity_db.get_entity_types(entity_id)
        candidate_type_strings = [f"{self.entity_db.get_entity_name(t)} ({t})"
                                  for t in candidate_types]
        description = self.entity_db.get_entity_description(entity_id)
        user_prompt = f"Entity: \"{entity_name}\" ({entity_id}), description: \"{description}\", list of types: [{', '.join(candidate_type_strings)}]"
        trial_num = 0
        response = None
        # Sleep for 1 second before querying GPT to avoid running into the rate limit per minute
        time.sleep(1)
        while trial_num < 3 and not response:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ])
            except openai.error.ServiceUnavailableError:
                print(f"Error trying to access the OpenAI API. Trying again...")
                trial_num += 1
                time.sleep(trial_num * 5)
        predicted_type_ids = None
        if response.choices:
            prediction_string = response.choices[0].message.content
            print(f"Prediction string: {prediction_string}")
            predicted_type_ids = get_qids(prediction_string)
            print(f"Predicted type IDs: {predicted_type_ids}")
        return predicted_type_ids
