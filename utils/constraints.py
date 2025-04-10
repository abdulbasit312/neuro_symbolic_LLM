from pysdd.sdd import SddManager, Vtree, WmcManager
from typing import List
from itertools import product
import torch

EPS = 1e-10

class ConstraintManager():

    def __init__(self, n_variables=2):
        self.n_variables = n_variables
        self.CONSTRAINT2FUNCTION = {
            "implication": self.implication_constraint,
            "negation": self.negation_constraint,
            "inverse_implication": self.inverse_implication_constraint,
            "all": self.all_constraints
        }

    @staticmethod
    def need_negations(constraint):
        """ Checks constraint type to split cases by design """
        return constraint in ["negation", "all"]

    def implication_constraint(self, l1:object, l2:object):
        """ Given sdd literals, return the formula """
        return (-l1 | l2)

    def inverse_implication_constraint(self, l1:object, l2:object):
        """ Given sdd literals, return the formula """
        return (l2 | -l1)

    def negation_constraint(self, l1:object, l2:object, l1_not:object, l2_not:object):
        """ Given sdd literals, return the formula """
        return ((l1 | l1_not) & (-(l1 & l1_not))) & ((l2 | l2_not) & (-(l2 & l2_not)))
        
    def all_constraints(self, l1:object, l2:object, l1_not:object, l2_not:object):
        """ Combined the constraints above"""
        return (
            self.implication_constraint(l1, l2) &
            self.negation_constraint(l1, l2, l1_not, l2_not) &
            self.inverse_implication_constraint(l1, l2)
        )

    def grounded_constraint(self, literals:List[int], ground_labels:List[int], constraint:str):
        """ Compile the constraint formula for the solver to compute the valid worlds
            literals:       List[int]                               from the constraint, signs for the literals
            ground_labels:  dict                                    ground truth assignments to constraint literals
            constraint:     str                                     constraint type
        """
        # Setup SDD and variables
        if ConstraintManager.need_negations(constraint): self.n_variables = 4
        else: self.n_variables = 2
        a_symbol, b_symbol = literals
        # Compile formula
        sdd = SddManager.from_vtree(Vtree(var_count=self.n_variables, var_order=list(range(1, self.n_variables+1)), vtree_type="balanced"))
        if ConstraintManager.need_negations(constraint): a, b, not_a, not_b = sdd.vars
        else: a, b = sdd.vars
        # Apply constraint negations if present in formula
        if a_symbol == 0: a = -a
        if b_symbol == 0: b = -b
        # Get constraint formula
        if ConstraintManager.need_negations(constraint): 
            if not_a is None or not_b is None: raise Exception("Invalid not_A or not_B probability tensor.") 
            formula = self.CONSTRAINT2FUNCTION[constraint](l1=a, l2=b, l1_not=not_a, l2_not=not_b)
        else: formula = self.CONSTRAINT2FUNCTION[constraint](l1=a, l2=b)
        # Apply grounding if provided
        if ground_labels:
            ant, cons = ground_labels
            # since a,b have been flipped before (symbol == 0), 
            # re-flip to take the grounding
            if ant == 0: 
                if a_symbol == 0: formula = formula & a
                else: formula = formula & -a
            if ant == 1: 
                if a_symbol == 0: formula = formula & -a
                else: formula = formula & a
            if cons == 0: 
                if b_symbol == 0: formula = formula & b
                else: formula = formula & -b
            if cons == 1: 
                if b_symbol == 0: formula = formula & -b
                else: formula = formula & b
        return formula
    
    def sl(self, p1:object, p2:object, batch_symbols:List[List[int]], batch_facts:List[List[int]], constraint:str, p1_not:object=None, p2_not:object=None):
        """ Compute the semantic loss from probabilities, constraint literals signs, constraint type 
            p:          (B, 2) tensor of batch probabilities for vars in a formula
            symbols:    (B, 2) tensor of symbols for vars in a formula  
            facts:      (B, 2) tensor of ground facts assignments for vars in a formula  
        """
        def generate_missing_assignments(d, keys):
            """ Augment implicit solver assignments (make explicit) """
            missing_keys = set(keys) - set(d.keys())
            new_dicts = []
            if not missing_keys: return [d]
            for combo in product([0, 1], repeat=len(missing_keys)):
                new_dict = d.copy()
                new_dict.update(dict(zip(missing_keys, combo)))
                new_dicts.append(new_dict)
            return new_dicts
        # select from batch_p of: B, 4, 1
        # where 4: p1, p2, p1_not, p2_not
        if ConstraintManager.need_negations(constraint):
            probabilities = torch.concat((p1, p2, p1_not, p2_not), dim=1) # B, 4, 1
        else: # normal ordering for the other
            probabilities = torch.concat((p1, p2), dim=1) # B, 2, 1
        # Apply sample-wise semantic loss and concatenate in a batch
        batch_loss = None
        for idx in range(probabilities.shape[0]):
            p = probabilities[idx] # 4, 1 or 2, 1
            s = batch_symbols[idx] # 2, 1
            g = None if batch_facts is None else batch_facts[idx] # B, 2, 1
            # get truth assignments and enumerate them if a subset is returned (works for 2 vars)
            models = [m for m in (self.grounded_constraint(literals=s, ground_labels=g, constraint=constraint)).models()]
            # Check and augment if keys are missing
            aug_models = []
            for m in models: aug_models.extend(generate_missing_assignments(d=m, keys=list(range(1, 5) if ConstraintManager.need_negations(constraint) else range(1, 3))))
            # Sum over possible worlds
            p_truth = []
            for model in aug_models:
                mask = ~torch.Tensor(list(model.values())).type(torch.bool)
                p_m = p.clone()
                p_m[mask] = 1 - p_m[mask]
                p_truth.append(p_m.prod())
            loss = -torch.log((torch.stack(p_truth).sum() + EPS)).unsqueeze(-1)
            if batch_loss is None: batch_loss = loss
            else: batch_loss = torch.concat((batch_loss, loss), dim=-1)
        return batch_loss.sum()