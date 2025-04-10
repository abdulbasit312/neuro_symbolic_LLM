import os
import json
import requests
from tqdm import tqdm

API_BASE = "https://api.conceptnet.io/c/en"
LIMIT = 800
entity = "object"
entities = ["human", "dog", "cat", "mammal", "car", "boat"]

# assumption:   a sub entity is a type of entity
# rationale:    get constraints for all general entities, 
#               get sub entities, 
#               test these constraints on the sub entities.
#               So training constraints will be on general entities,
#               training facts will be just (entity, IsA, entity)
#               test facts will be just     (sub_entity, IsA, entity)
#               which triggers the constraints

constraints = {"links": []}
train_facts = {}
test_facts = {}
all_facts = {}

# Construct train facts and constraints
for entity in tqdm(entities):
    url = f"{API_BASE}/{entity}?offset=0&limit={LIMIT}"
    obj = requests.get(url).json()
    train_facts[entity] = {f"IsA,{entity}": "yes"}
    all_facts[entity] = {f"IsA,{entity}": "yes"}
    # Dumping
    for edge in obj["edges"]:
        link = {
            "weight": "yes_yes",
            "direction": "forward",
        }
        # CapableOf
        if edge["rel"]["label"] == "CapableOf":
            link["source"] = f"IsA,{entity}"
            link["target"] = f"CapableOf,{edge['end']['label'].replace(',', ' ')}"
            constraints["links"].append(link)
            all_facts[entity][f"CapableOf,{edge['end']['label'].replace(',', ' ')}"] = "yes"

        # AtLocation
        elif edge["rel"]["label"] == "AtLocation":
            # entity is Y
            if f"/c/en/{entity}" in edge["end"]["@id"]:
                link["source"] = f"IsA,{entity}"
                link["target"] = f"HasA,{edge['start']['label'].replace(',', ' ')}"
                constraints["links"].append(link)
                all_facts[entity][f"HasA,{edge['start']['label'].replace(',', ' ')}"] = "yes"
        # IsA
        elif edge["rel"]["label"] == "IsA":
            # X is entity
            if f"/c/en/{entity}" in edge["start"]["@id"]:
                link["source"] = f"IsA,{entity}"
                link["target"] = f"HasA,{edge['end']['label'].replace(',', ' ')}"
                constraints["links"].append(link)
                all_facts[entity][f"HasA,{edge['end']['label'].replace(',', ' ')}"] = "yes"

print(f"train_facts: {len([k for e in train_facts for k in train_facts[e].keys()])}")
print(f"constraints: {len(constraints['links'])}")

# Construct test facts
for super_entity in tqdm(entities):
    url = f"{API_BASE}/{super_entity}?offset=0&limit={LIMIT}"
    obj = requests.get(url).json()
    for edge in obj["edges"]:
        # IsA
        if edge["rel"]["label"] == "IsA":
            # X is entity
            if f"/c/en/{super_entity}" in edge["end"]["@id"]:
                sub_entity = edge['start']['label']
                test_facts[sub_entity] = {f"IsA,{super_entity}": "yes"}

print(f"test_facts: {len([k for e in test_facts for k in test_facts[e].keys()])}")

# Writing to files
destination_path = os.path.join("data", "conceptnet")
with open(os.path.join(destination_path, "train_facts.json"), "w") as f:
    json.dump(train_facts, f)
    f.close()
with open(os.path.join(destination_path, "all_facts.json"), "w") as f:
    json.dump(all_facts, f)
    f.close()
with open(os.path.join(destination_path, "test_facts.json"), "w") as f:
    json.dump(test_facts, f)
    f.close()
with open(os.path.join(destination_path, "constraints.json"), "w") as f:
    json.dump(constraints, f)
    f.close()