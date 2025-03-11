from .utils import TorchDataset
from torch.utils.data import DataLoader
from typing import List
import random
import json
import os

DATA_WORKERS = 3

class ConceptNet:

    def __init__(self):
        self.templates = os.path.join("data", "beliefbank", "templates.json")
        self.uncountables = os.path.join("data", "beliefbank", "non_countable.txt")

    """ Dataloaders """
    def get_dataloaders(self, batch_size:int, constraints_path:str, dist_1_facts_path:str, dist_2_facts_path:str, all_facts_path:str):
        constraints_links = self.get_constraints_links(path=constraints_path)
        # Facts
        all_facts = self.get_facts_all(path=all_facts_path)
        dist_1_facts_all = self.get_facts_all(path=dist_1_facts_path)
        dist_2_facts_all = self.get_facts_all(path=dist_2_facts_path)
        # Constraints
        constraints_all = TorchDataset(samples=constraints_links)
        constraints_clb_train = TorchDataset(
            samples=self.get_grounded_constraints(facts=dist_1_facts_all, path=constraints_path))
        # Dataloaders
        return {
            "constraints": {
                "train": DataLoader(dataset=constraints_clb_train, batch_size=batch_size, shuffle=False, num_workers=DATA_WORKERS),
                "val": DataLoader(dataset=constraints_all, batch_size=batch_size, shuffle=False, num_workers=DATA_WORKERS),
                "test": DataLoader(dataset=constraints_all, batch_size=batch_size, shuffle=False, num_workers=DATA_WORKERS),
            },
            "facts": {
                "train": {
                    "dist_1": DataLoader(
                        dataset=TorchDataset(samples=dist_1_facts_all), 
                        batch_size=batch_size, shuffle=False, num_workers=DATA_WORKERS)
                },
                "test": {
                    "dist_2": DataLoader(
                        dataset=TorchDataset(samples=dist_2_facts_all), 
                        batch_size=batch_size, shuffle=False, num_workers=DATA_WORKERS),
                },
                "all": DataLoader(
                        dataset=TorchDataset(samples=all_facts), 
                        batch_size=batch_size, shuffle=False, num_workers=DATA_WORKERS),
            }
        }

    """ Constraints """
    def get_grounded_constraints(self, path:str, facts:List) -> List[object]:
        """ Istantiate links into constraints for each subject (using a fact split)
            Returns:
                samples:List[object]    {antecedent, consequent, s_antecedent, s_consequent, g_antecedent, g_consequent} 
        """
        # Facts list -> hash table
        hash_facts = dict()
        for fact in facts: 
            hash_facts.setdefault(fact["predicate"], dict())[fact["subject"]] = fact
        # NL formatting
        templates, uncountables = self.get_language_templates(
            templates_path=self.templates, 
            uncountables_path=self.uncountables
        )
        general_constraints = self.get_constraints_links(path)
        samples = []
        for constraint in general_constraints: # for each principle
            # check grounding
            # for also ungrounded constraints: else: ant_grounding = -1
            if constraint["ant"] in hash_facts.keys():
                # 1 predicate -> n subjects, n constraints
                for subj, belief in hash_facts[constraint["ant"]].items():
                    ant_grounding = hash_facts[constraint["ant"]][subj]["answer"]
                    rel, obj = constraint["cons"].split(",") # extract the implied consequent
                    cons = self.get_implication_string(
                        templates=templates, uncountables=uncountables, 
                        subject=subj, relation=rel, 
                        symbol=True, obj=obj
                    )
                    cons_negated = self.get_implication_string(
                        templates=templates, uncountables=uncountables, 
                        subject=subj, relation=rel, 
                        symbol=False, obj=obj
                    )
                    sample = {
                        "ant": belief["fact"], 
                        "ant_negated": belief["negated_fact"], 
                        "cons": cons,
                        "cons_negated": cons_negated,
                        "ant_sign": int(constraint["ant_sign"]), 
                        "cons_sign": int(constraint["cons_sign"]),
                        "ant_grounding": ant_grounding
                    }
                    samples.append(sample)
        return samples

    def get_constraints_links(self, path) -> (List[object]):
        """ Load all logical constraints and index by source edge 
            Parameters:
                path:str            Path to the constraints file (json expected)
            Returns:
                links:List[obj]     List of {antecedent, consequent, symbol_ant, symbol_conseq}
        """
        with open(path) as f:
            constraints = json.load(f)
        links = []
        for rel in constraints["links"]:
            if rel["direction"] == "forward":
                links.append({
                    "ant": rel["source"], 
                    "cons": rel["target"], 
                    "ant_sign": rel["weight"].split("_")[0] == "yes", 
                    "cons_sign": rel["weight"].split("_")[1] == "yes"
                })
        return links  

    """ Facts """
    def get_facts_all(self, path:str):
        """  Convert the BeliefBank facts into a set of NL samples """
        templates, uncountables = self.get_language_templates(
            templates_path=self.templates, 
            uncountables_path=self.uncountables
        )
        with open(path) as f:
            facts = json.load(f) 
            samples = []
            for subject, subject_facts in facts.items():
                for key, belief in subject_facts.items():
                    relation, obj = key.split(",")
                    fact = self.get_implication_string(
                        templates=templates, uncountables=uncountables, 
                        subject=subject, relation=relation, 
                        symbol=True, obj=obj
                    )
                    negated_fact = self.get_implication_string(
                        templates=templates, uncountables=uncountables, 
                        subject=subject, relation=relation, 
                        symbol=False, obj=obj
                    )
                    samples.append({
                        "subject": subject, 
                        "predicate": key, 
                        "fact": fact, 
                        "negated_fact": negated_fact, 
                        "answer": int(belief == "yes")
                    })
        return samples

    """ Language formatting methods """
    def get_implication_string(self, templates, uncountables, subject, relation, symbol, obj):
        """ From implication object, convert into natural relation"""
        this_template = templates[relation]
        X = self.get_fluent_noun(subject, uncountables)
        Y = self.get_fluent_noun(obj, uncountables, relation)
        rng = random.randint(0, 1)
        # questions: templates, templates_negated
        # statements: assertion_positive, assertion_negative
        if symbol: nl_question = this_template['assertion_positive'].format(X=X, Y=Y)
        else: nl_question = this_template['assertion_negative'].format(X=X, Y=Y)
        return nl_question

    def get_language_templates(self, templates_path, uncountables_path):
        """ Read helper structures from file"""
        # from https://github.com/eric-mitchell/concord
        with open(templates_path) as f:
            natural_relations = json.load(f) 
            f.close()
        with open(uncountables_path) as f:
            uncountables = f.read().split('\n')
            f.close()
        return natural_relations, uncountables

    def get_fluent_noun(self, noun, uncountables, relation=None):
        """
        Make a noun phrase 'fluenter' (more fluent) before putting it in a
        template.  note we only a.) check if the noun is in a list of known
        non-countables or has a relation with a certain type, and b.) look at the
        first letter of the input to decide whether to put a or an.

        :param noun: the noun (phrase) -- subject or object -- to make more fluent
        :param uncountables: the list of uncountables to compare to
        :param relation: BeliefBank relation
        :return: a string with the prettified noun phrase
        """
        # from https://github.com/eric-mitchell/concord
        if noun in uncountables:
            return noun
        if relation is not None:
            if relation in ['CapableOf', 'MadeOf', 'HasProperty']:
                return noun
        if noun[0] in ['a', 'e', 'i', 'o', 'u']:
            return 'an ' + noun
        return 'a ' + noun