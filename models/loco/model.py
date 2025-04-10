import re
import transformers
import torch
import warnings
import torch.nn as nn
from typing import Callable
import random
from typing import List
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from .prompts import *
from utils.training import *

EPS = 1e-10

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class QA(nn.Module):

    def __init__(
        self,
        factuality,
        use_table_truth,
        gpu_id,
        model_hf_name="allenai/macaw-large",
        quantization: bool = False,
        in_format: Callable = None,
        out_format: Callable = None,
        mod_factory: Callable = None,
        tok_factory: Callable = None,
    ):
        super(QA, self).__init__()

        self.model_hf_name = model_hf_name
        self.factuality = factuality
        self.use_table_truth = use_table_truth

        hf_models = {
            "allenai/macaw-large": {
                "type": "seq2seq"
            },
            "allenai/macaw-3b": {
                "type": "seq2seq"
            },
            "google/t5-small-ssm-nq": {
                "type": "seq2seq"
            },
            "google/t5-large-ssm-nq": {
                "type": "seq2seq"
            },
            "google/flan-t5-small": {
                "type": "seq2seq"
            },
            "google/t5-3b-ssm-nq": {
                "type": "seq2seq"
            },
            "unc-nlp/lxmert-base-uncased": {
                "type": "seq2seq"
            },
            "gpt2-medium": {
                "type": "decoder"
            },
            "sharpbai/Llama-2-7b-chat": {
                "type": "decoder"
            },
            "sharpbai/Llama-2-7b-hf": {
                "type": "decoder"
            },
            "sharpbai/Llama-2-13b-hf": {
                "type": "decoder"
            },
            "NousResearch/Llama-2-7b-chat-hf": {
                "type": "decoder"
            },
            "mistralai/Mistral-7B-Instruct-v0.1": {
                "type": "decoder"
            },
            "NousResearch/Llama-2-70b-chat-hf": {
                "type": "decoder"
            },
            "NousResearch/Llama-2-7b-hf": {
                "type": "decoder"
            },
            "NousResearch/Llama-2-70b-hf": {
                "type": "decoder"
            },
            "TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T": {
                "type": "decoder"
            },
            "NousResearch/Llama-2-13b-hf": {
                "type": "decoder"
            },
            "meta-llama/Llama-3.2-3B": {
                "type": "decoder"
            },
        }
        self.hf_models = hf_models

        hf_mod_config = {
            "allenai/macaw-large": (
                in_format if in_format else in_f_macaw,
                out_format if out_format else out_f_macaw,
                transformers.AutoTokenizer.from_pretrained,
                transformers.AutoModelForSeq2SeqLM.from_pretrained,
            ),
            "allenai/macaw-3b": (
                in_format if in_format else in_f_macaw,
                out_format if out_format else out_f_macaw,
                transformers.AutoTokenizer.from_pretrained,
                transformers.AutoModelForSeq2SeqLM.from_pretrained,
            ),
            "google/t5-small-ssm-nq": (
                lambda x: x,
                lambda x: x,
                transformers.AutoTokenizer.from_pretrained,
                transformers.T5ForConditionalGeneration.from_pretrained,
            ),
            "google/flan-t5-small": (
                in_format if in_format else in_f_flant5,
                lambda x: x,
                transformers.AutoTokenizer.from_pretrained,
                transformers.AutoModelForSeq2SeqLM.from_pretrained,
            ),
            "gpt2-medium": (
                in_format if in_format else in_f_flant5,
                lambda x: x,
                transformers.AutoTokenizer.from_pretrained,
                transformers.AutoModelForCausalLM.from_pretrained,
            ),
            "google/t5-large-ssm-nq": (
                lambda x: x,
                lambda x: x,
                transformers.AutoTokenizer.from_pretrained,
                transformers.AutoModelForSeq2SeqLM.from_pretrained,
            ),
            "google/t5-3b-ssm-nq": (
                lambda x: x,
                lambda x: x,
                transformers.AutoTokenizer.from_pretrained,
                transformers.AutoModelForSeq2SeqLM.from_pretrained,
            ),
            "sharpbai/Llama-2-7b-chat": (
                lambda x: x,
                lambda x: x,
                transformers.AutoTokenizer.from_pretrained,
                transformers.AutoModelForCausalLM.from_pretrained,
            ),
            "sharpbai/Llama-2-7b-hf": (
                lambda x: x,
                lambda x: x,
                transformers.AutoTokenizer.from_pretrained,
                transformers.AutoModelForCausalLM.from_pretrained,
            ),
            "NousResearch/Llama-2-7b-chat-hf": (
                lambda x: x,
                lambda x: x,
                transformers.AutoTokenizer.from_pretrained,
                transformers.AutoModelForCausalLM.from_pretrained,
            ),
            "sharpbai/Llama-2-13b-hf": (
                lambda x: x,
                lambda x: x,
                transformers.AutoTokenizer.from_pretrained,
                transformers.AutoModelForCausalLM.from_pretrained,
            ),
            "NousResearch/Llama-2-70b-chat-hf":(
                lambda x: x,
                lambda x: x,
                transformers.AutoTokenizer.from_pretrained,
                transformers.AutoModelForCausalLM.from_pretrained,
            ),
            "mistralai/Mistral-7B-Instruct-v0.1": (
                lambda x: x,
                lambda x: x,
                transformers.AutoTokenizer.from_pretrained,
                transformers.AutoModelForCausalLM.from_pretrained,
            ),
            "NousResearch/Llama-2-7b-hf": (
                lambda x: x,
                lambda x: x,
                transformers.AutoTokenizer.from_pretrained,
                transformers.AutoModelForCausalLM.from_pretrained,
            ),
            "NousResearch/Llama-2-70b-hf": (
                lambda x: x,
                lambda x: x,
                transformers.AutoTokenizer.from_pretrained,
                transformers.AutoModelForCausalLM.from_pretrained,
            ),
            "TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T": (
                lambda x: x,
                lambda x: x,
                transformers.AutoTokenizer.from_pretrained,
                transformers.AutoModelForCausalLM.from_pretrained,
            ),
            "NousResearch/Llama-2-13b-hf": (
                lambda x: x,
                lambda x: x,
                transformers.AutoTokenizer.from_pretrained,
                transformers.AutoModelForCausalLM.from_pretrained,
            ),
            "meta-llama/Llama-3.2-3B": (
                lambda x: x,
                lambda x: x,
                transformers.AutoTokenizer.from_pretrained,
                transformers.AutoModelForCausalLM.from_pretrained,
            ),
        }

        if model_hf_name not in hf_models.keys():
            self.in_format = lambda x: x
            self.out_format = lambda x: x
            self.tok_factory = tok_factory
            self.mod_factory = mod_factory

            if any(
                factory is None
                for factory in [self.tok_factory, self.mod_factory]
            ):
                warnings.warn(
                    f"""
                    Behaviour may be undefined if model_hf_name is not in
                    {hf_models.keys()}, or mod_factory and tok_factory are unset.
                    
                    Defaults are:
                        mod: transformers.AutoModelForSeq2SeqLM.from_pretrained
                        tok: transformers.AutoTokenizer.from_pretrained
                        
                    If these defaults are correct this warning can be safely 
                    ignored.
                """
                )
        else:
            (
                self.in_format,
                self.out_format,
                self.tok_factory,
                self.mod_factory,
            ) = hf_mod_config[model_hf_name]

        self.gpu_id = gpu_id

        # Model
        self.tokenizer = self.tok_factory(self.model_hf_name) if "t5-3b" not in self.model_hf_name \
            else self.tok_factory("google/t5-xl-ssm-nq")

        if self.is_decoder():
            if quantization:
                print("[-] Loading decoder only model with quantization")

                # Model configuration
                self.model = self.mod_factory(
                    self.model_hf_name,
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type='nf4'
                    ),
                    torch_dtype=torch.bfloat16,
                    cache_dir="/w/331/svajpayee/neuro_symbolic_LLM/assets"
                )
                self.model.config.use_cache = False
                self.model = prepare_model_for_kbit_training(self.model)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = "right"
                peft_config = LoraConfig(
                    r=128,
                    lora_alpha=16,
                    target_modules=find_all_linear_names(self.model),
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                self.model = get_peft_model(self.model, peft_config)
                print_trainable_parameters(self.model)
            else:
                print("[-] Loading decoder only model")
                self.model = self.mod_factory(
                    self.model_hf_name
                )
        elif self.is_seq2seq():
            print("[-] Loading seq2seq model")
            self.model = self.mod_factory(
                self.model_hf_name,
                torch_dtype=torch.bfloat16,
            )
        else: raise Exception("Invalid model type. It must be either decoder-only or seq2seq.")

        # Fix padding tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.MAX_GEN_TOKENS = 4
        
    def get_model_type(self):
        return self.hf_models[self.model_hf_name]["type"]

    def is_decoder(self):
        return self.hf_models[self.model_hf_name]["type"] == "decoder"

    def is_seq2seq(self):
        return self.hf_models[self.model_hf_name]["type"] == "seq2seq"
        
    def infer_fact(self, quests:List[str]):
        """ Get model answers to (premise, hypothesis) formulas """
        self.model.eval()
        in_prompts = self.tokenizer(quests, padding=True, return_tensors="pt").to(self.gpu_id)
        answers = self.tokenizer.batch_decode(self.model.generate(**in_prompts, do_sample=True, top_k = 50, top_p = 1.0, temperature = 1.0, max_new_tokens = self.MAX_GEN_TOKENS), skip_special_tokens=True)
        return answers

    def infer_fact_prob(self, quests:List[str]):
        """ Get single fact probabilities """
        self.model.eval()
        if self.is_seq2seq():
            return self.__seq2seq_infer_fact_prob(quests)

        elif self.is_decoder():
            return self.__decoder_infer_fact_prob(quests)

    def __decoder_infer_fact_prob(self, quests:List[str]):
        """ Get single fact probabilities """
        self.model.eval()
        targets_sents = [FALSETRUE[1] for idx in range(len(quests))]
        return gpt_get_target_probs(
            model=self.model, 
            tokenizer=self.tokenizer, 
            inputs=quests,
            targets=targets_sents,
            gpu_id=self.gpu_id
        )

    def __seq2seq_infer_fact_prob(self, quests:List[str]):
        """ Get single fact probabilities """
        self.model.eval()
        positive_labels = [prompt_answer(self.get_model_type(), FALSETRUE[1]) for idx in range(len(quests))]
        positive_answ = self.tokenizer(positive_labels, padding=True, return_tensors="pt")
        positive_masked_output = positive_answ.input_ids.masked_fill(positive_answ.input_ids == self.tokenizer.pad_token_id, -100)
        in_premise = self.tokenizer(quests, padding=True, return_tensors="pt")
        outputs = self.model(input_ids=in_premise.input_ids.to(self.gpu_id), labels=positive_answ.input_ids.to(self.gpu_id))
        probs, _ = seq2seq_get_target_probs(outputs.logits.to(self.gpu_id), positive_masked_output.to(self.gpu_id), should_reduce=False) 
        probs = probs.exp().unsqueeze(-1)
        return probs 

    def get_facts_loss(self, statements:List[str], labels:List[int]):
        """ Train: fine-tuning on ground single facts """
        self.model.train()
        ground_labels = [prompt_answer(type=self.get_model_type(), text=FALSETRUE[int(l)]) for l in labels]
        return - torch.sum(
                    torch.log(
                        gpt_get_target_probs(
                            model=self.model, 
                            tokenizer=self.tokenizer, 
                            inputs=statements,
                            targets=ground_labels,
                            gpu_id=self.gpu_id
                        )
                    ))
    
    def get_formula_beliefs(self, s1, s2, label=None):
        """ 
            Train: fine-tuning on whole prompts
            Inputs:
                s1                List[str]   formatted antecedent facts prompts
                s2              List[str]   formatted consequent facts prompts 
            Outputs:
                s1_probs           Tensor      LM probabilities of ant. facts
                s2_probs        Tensor      LM probabilities of cons. facts
                cond_s2_probs   Tensor      LM probabilities of cons. facts conditioned on antecedents (in-context assumption)
        """
        self.model.train()
        
        if self.is_seq2seq():
            return self.__seq2seq_get_formula_beliefs(s1, s2, label)

        elif self.is_decoder():
            return self.__decoder_get_formula_beliefs(s1, s2, label)

    def __seq2seq_get_formula_beliefs(self, s1, s2, label=None):
        """ 
            Train: fine-tuning on whole prompts
            Inputs:
                s1                List[str]   formatted antecedent facts prompts
                s2              List[str]   formatted consequent facts prompts 
            Outputs:
                s1_probs           Tensor      LM probabilities of ant. facts
                s2_probs        Tensor      LM probabilities of cons. facts
                cond_s2_probs   Tensor      LM probabilities of cons. facts conditioned on antecedents (in-context assumption)
        """
        if label is None: label = FALSETRUE[1]
        positive_labels = [prompt_answer(self.get_model_type(), label) for idx in range(len(s1))]
        positive_answ = self.tokenizer(positive_labels, padding=True, return_tensors="pt")
        positive_masked_output = positive_answ.input_ids.masked_fill(
            positive_answ.input_ids == self.tokenizer.pad_token_id, -100
        )

        # ---- Factual: s1
        in_s1 = self.tokenizer(s1, padding=True, return_tensors="pt")
        outputs = self.model(input_ids=in_s1.input_ids.to(self.gpu_id), labels=positive_answ.input_ids.to(self.gpu_id))
        s1_probs, _ = seq2seq_get_target_probs(outputs.logits.to(self.gpu_id), positive_masked_output.to(self.gpu_id), should_reduce=False) 
        s1_probs = s1_probs.exp().unsqueeze(-1)

        # ---- Factual: s2
        in_s2 = self.tokenizer(s2, padding=True, return_tensors="pt")
        outputs = self.model(input_ids=in_s2.input_ids.to(self.gpu_id), labels=positive_answ.input_ids.to(self.gpu_id),)
        s2_probs, _ = seq2seq_get_target_probs(outputs.logits.to(self.gpu_id), positive_masked_output.to(self.gpu_id), should_reduce=False) 
        s2_probs = s2_probs.exp().unsqueeze(-1)  

        return s1_probs, s2_probs

    def __decoder_get_formula_beliefs(self, s1, s2, label=None):
        """ 
            Train: fine-tuning on whole prompts
            Inputs:
                s1                List[str]   formatted antecedent facts prompts
                s2              List[str]   formatted consequent facts prompts 
            Outputs:
                s1_probs           Tensor      LM probabilities of ant. facts
                s2_probs        Tensor      LM probabilities of cons. facts
                cond_s2_probs   Tensor      LM probabilities of cons. facts conditioned on antecedents (in-context assumption)
        """
        if label is None: label = FALSETRUE[1]
        positive_labels = [prompt_answer(self.get_model_type(), label) for idx in range(len(s1))]
        # ---- Factual: s1
        s1_probs = gpt_get_target_probs(
            model=self.model, 
            tokenizer=self.tokenizer, 
            inputs=s1,
            targets=positive_labels,
            gpu_id=self.gpu_id
        )
        # ---- Factual: s2
        s2_probs = gpt_get_target_probs(
            model=self.model, 
            tokenizer=self.tokenizer, 
            inputs=s2,
            targets=positive_labels,
            gpu_id=self.gpu_id
        )
        # ---- SL: conditioned s2
        # assuming the s1 is always to true
        cond_hypoths = [f"{p} {positive_labels[idx]} {s2[idx]}" for idx, p in enumerate(s1)]
        cond_s2_probs = gpt_get_target_probs(
            model=self.model, 
            tokenizer=self.tokenizer, 
            inputs=cond_hypoths,
            targets=positive_labels,
            gpu_id=self.gpu_id
        )
        return s1_probs, s2_probs

    def get_perplexity(self, data, window_size=512):
        """ Computes perplexity as in https://huggingface.co/docs/transformers/en/perplexity """
        encodings = self.tokenizer("\n\n".join(data), return_tensors="pt")
        # model configs
        max_length = self.model.config.max_length
        seq_len = encodings.input_ids.size(1)
        nlls = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, window_size)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.gpu_id)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss
            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        return torch.exp(torch.stack(nlls).mean()).item()
        

def gpt_get_target_probs(model, tokenizer, inputs, targets, gpu_id):
    """ Token by token prediction (greedy sampling) and probs given target """
    # B = batch size, L = seq.len, D = dict size
    input_ids = tokenizer(inputs, padding=True, return_tensors="pt").to(gpu_id).input_ids # B, L
    target_ids = tokenizer(targets, padding=True, return_tensors="pt").to(gpu_id).input_ids # B, L
    target_toks = target_ids[:, 1:]
    # answers = model.generate(input_ids, max_new_tokens=target_toks.shape[1])
    # print(tokenizer.batch_decode(answers))
    probs = None
    # forward() automatically shifts by 1 and computes loss
    logits = model(input_ids=input_ids).logits.softmax(-1) # B, L, D
    # gather last token logits with the target ids
    target_probs = torch.gather(logits[:, -1], 1, target_toks[:, 0].unsqueeze(-1))
    return target_probs

def gpt_get_seq_target_probs(model, tokenizer, inputs, targets, gpu_id):
    """ Token by token prediction (greedy sampling) and probs given target """
    input_ids = tokenizer(inputs, padding=True, return_tensors="pt").to(gpu_id).input_ids # B, L
    target_ids = tokenizer(targets, padding=True, return_tensors="pt").to(gpu_id).input_ids # B, L
    target_toks = target_ids[:, 1:]
    probs = None
    for idx in range(target_toks.shape[1]):
        logits = model(input_ids=input_ids).logits.log_softmax(-1) # B, L, D
        # forward() automatically shifts by 1 and computes loss
        input_ids = torch.hstack((input_ids, logits[:, -1].argmax(-1).unsqueeze(-1))) # B, L, 1
        if probs is None: probs = target_probs
        else: probs = torch.hstack((probs, target_probs))
    return probs.sum(-1)

def mask_hf_labels(labels, null_token=0):
    valid_mask = labels != -100
    valid_labels = labels.masked_fill(~valid_mask, null_token)
    return valid_mask, valid_labels

def gather_log_probs(logits, labels):
    assert labels.dim() == logits.dim() - 1
    assert labels.shape == logits.shape[:-1]
    # softmax over dictionary size, labels are the indices (takes an array of these probabilities)
    return logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

def seq2seq_get_target_probs(pred, targets, should_reduce=True):
    NULL_TOKEN = 0  # a placeholder used for masked target locations

    pred = pred.clone()
    # mask out padding tokens
    mask, targ = mask_hf_labels(targets)
    # over the dictionary, take only the generated tokens' probabilities
    unmasked_log_probs = gather_log_probs(pred, targ)

    pred_ids = pred.argmax(-1).masked_fill(~mask, NULL_TOKEN)
    correct = pred_ids == targ
    if pred.dim() == 3:
        correct = (pred_ids == targ).all(
            -1
        )  # We want to get the whole sequence right
    acc = correct.float()

    if should_reduce:
        acc = acc.mean()

    # default: no mean reduction
    if should_reduce:
        log_probs = (unmasked_log_probs * mask.float()).mean(-1)
    else:
        log_probs = (unmasked_log_probs * mask.float()).sum(-1)
        
    return log_probs, acc
