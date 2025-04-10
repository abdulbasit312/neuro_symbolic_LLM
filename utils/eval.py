from typing import List
from pysdd.sdd import SddManager, Vtree

class Metrics:

    @staticmethod
    def multihop_consistency(model_beliefs:dict, ref_beliefs:dict, constraints:List[object]) -> object:
        """
            Compute consistency on the whole set of links
            beliefs:dict            {fact:str: answer:int(yes/no)}
            ref_beliefs:dict        {fact:str: answer:int(yes/no)}
        """
        antecedents = [ant for l in constraints for ant in l["antecedent"]]
        consequents = [cons for l in constraints for cons in l["consequent"]]
        ground_truth_values = [s for l in constraints for s in l["s_consequent"]]
        applicable = 0
        violated = 0
        for i in range(len(antecedents)):
            A = antecedents[i]
            B = consequents[i]
            for subj in ref_beliefs.keys():
                if B in antecedents: # B is a middle step in 2 hop ops
                    if (A in model_beliefs[subj]) and (B in model_beliefs[subj]): # model knows A, B
                        A_believed = (ref_beliefs[subj][A] == 1) and (model_beliefs[subj][A] == 1)
                        B_believed = (ref_beliefs[subj][B] == 1) and (model_beliefs[subj][B] == 1)
                        if A_believed and B_believed: # A, B are true
                            # get all Cs from: A -> B -> C
                            C_facts = [(jdx, consequents[jdx]) for jdx in range(len(antecedents)) if antecedents[jdx] == B]
                            for jdx, C in C_facts: # does C match?
                                if C in model_beliefs[subj]:
                                    ct = int(ground_truth_values[jdx].item())
                                    applicable += 1
                                    if ct != model_beliefs[subj][C]: violated += 1
        print(f"multi-hop consistency; applicable: {applicable}; violated: {violated}")
        return 1 - ((violated/applicable) if applicable > 0 else 1)

    @staticmethod
    def perplexity(model:object, inputs:List[str], outputs:List[str]) -> float:
        """
            Computes the perplexity in computing outputs given inputs tok by tok
            model:object            A wrapper to a pytorch model containing also the tokenizer
            inputs:List             Questions optionally with context
            outputs:List            Expected outputs
        """
        in_quests = model.tokenizer(inputs, padding=True, return_tensors="pt").to(model.gpu_id)
        in_answs = model.tokenizer(outputs, padding=True, return_tensors="pt").to(model.gpu_id)
        answers = model.tokenizer.batch_decode(model.model.generate(**in_quests, max_length=128), skip_special_tokens=True)
        outputs = model.model(input_ids=in_quests.input_ids, labels=in_answs.input_ids)
        return torch.exp(outputs.loss).item()
        
    @staticmethod
    def consistency(model_beliefs:dict, ref_beliefs:dict, constraints:List[object]) -> float:
        """
            Compute consistency on the whole set of links
            beliefs:dict            {fact:str: answer:int(yes/no)}
            ref_beliefs:dict        {fact:str: answer:int(yes/no)}
        """
        antecedents = [ant for l in constraints for ant in l["antecedent"]]
        consequents = [cons for l in constraints for cons in l["consequent"]]
        ground_truth_values = [s for l in constraints for s in l["s_consequent"]]
        applicable = 0
        violated = 0
        for i in range(len(antecedents)):
            ant = antecedents[i]
            cons = consequents[i]
            for subj in ref_beliefs.keys():
                # the model may not believe an antecedent that is true: we don't penalize it, factuality will do
                if (ant in model_beliefs[subj]) and (ant in ref_beliefs[subj]) and (ref_beliefs[subj][ant] == 1): # ref believes ant is true
                    ct = int(ground_truth_values[i].item()) # hypothesis according to the costraint
                    if (cons in model_beliefs[subj].keys()): # model has a belief about the hypothesis
                        applicable += 1
                        # print(f"ant: {model_beliefs[subj][ant]}, cons: {model_beliefs[subj][cons]}")
                        if ct != model_beliefs[subj][cons]: violated += 1
        print(f"consistency; applicable: {applicable}; violated: {violated}")
        return 1 - ((violated/applicable) if applicable > 0 else 1)

    @staticmethod
    def inverse_consistency(model_beliefs:dict, ref_beliefs:dict, constraints:List[object]) -> float:
        """
            Compute consistency on the whole set of links
            beliefs:dict            {fact:str: answer:int(yes/no)}
            ref_beliefs:dict        {fact:str: answer:int(yes/no)}
        """
        antecedents = [ant for l in constraints for ant in l["antecedent"]]
        consequents = [cons for l in constraints for cons in l["consequent"]]
        applicable = 0
        violated = 0
        for i in range(len(antecedents)):
            ant = antecedents[i]
            cons = consequents[i]
            for subj in ref_beliefs.keys():
                # the model may not believe an antecedent that is true: we don't penalize it, factuality will do
                if (cons in model_beliefs[subj]) and (cons in ref_beliefs[subj]) and (model_beliefs[subj][cons] == 0):
                    if (ant in model_beliefs[subj].keys()): # model has a belief about the hypothesis
                        applicable += 1
                        if model_beliefs[subj][ant] == 1: violated += 1 # violating "-B -> -A"?
        print(f"inverse consistency; applicable: {applicable}; violated: {violated}")
        return 1 - ((violated/applicable) if applicable > 0 else 1)
    
    @staticmethod
    def negation_consistency(beliefs:List[int], negated_beliefs:List[int]) -> float:
        violations = [int(b == n_b) for b, n_b in zip(beliefs, negated_beliefs)]
        print(f"negation consistency; applicable: {len(violations)}; violated: {sum(violations)}")
        return 1 - (sum(violations) / len(violations)) 

    @staticmethod
    def get_truth_table(symbols:List[bool], n_variables:int=2) -> (List, List):
        """Get sdd formula from vars"""
        sdd = SddManager.from_vtree(Vtree(var_count=n_variables, var_order=list(range(1, n_variables+1)), vtree_type="balanced"))
        a_symbol, b_symbol = symbols
        a, b = sdd.vars
        if a_symbol == False: a = -a
        if b_symbol == False: b = -b
        return ((-a | b).models()), ((-(-a | b)).models()) # T:List[], F:List[]

    @staticmethod
    def satisfiability(model_beliefs:dict, constraints:List[object]) -> float:
        """
            Compute satisfiability on the whole set of links
            beliefs:dict        {fact:str: answer:int(yes/no)}
        """
        antecedents = [ant for l in constraints for ant in l["antecedent"]]
        consequents = [cons for l in constraints for cons in l["consequent"]]
        ant_symbol = [s for l in constraints for s in l["s_antecedent"]]
        cons_symbol = [s for l in constraints for s in l["s_consequent"]]
        applicable = 0
        satisfied = 0
        for i in range(len(antecedents)):
            ant = antecedents[i]
            cons = consequents[i]
            for subj in model_beliefs.keys():
                if (ant in model_beliefs[subj]) and (cons in model_beliefs[subj]): # antecedent is believed for the subj
                    true_assignments, _ = Metrics.get_truth_table((ant_symbol[i], cons_symbol[i]))
                    true_assignments = [list(s.values()) for s in true_assignments]
                    model_assignment = [model_beliefs[subj][ant], model_beliefs[subj][cons]]
                    satisfied += int(model_assignment in true_assignments)
                    applicable += 1
        return ((satisfied/applicable) if applicable > 0 else 0)
