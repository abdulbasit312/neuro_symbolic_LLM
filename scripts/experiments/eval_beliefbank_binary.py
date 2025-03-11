"""
    Diego Calanzone, 2024 University of Trento
    Experiment 8:
        Qualitative analysis of loco-llama-2-7b outputs
"""
import transformers
import json
import os
from models.loco.prompts import *
from typing import List
import torch
from datasets import load_dataset
import random
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from utils.training import *
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

random.seed(1337)

def make_batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def get_beliefs(model, tokenizer, quests:List[str], maxtok=4):
    """ Get model answers to (premise, hypothesis) formulas """
    model.eval()
    in_prompts = tokenizer(quests, padding=True, return_tensors="pt").to(device)
    answers = tokenizer.batch_decode(model.generate(**in_prompts, max_new_tokens = maxtok), skip_special_tokens=True)
    # answers = [parse_answer(a) for a in answers]
    return answers

def get_facts_entailmentbank(path):
    d_entailmentbank = [json.loads(s) for s in list(open(path))]
    facts = []
    for d in d_entailmentbank:
        facts += list(d["meta"]["triples"].values()) + list(d["meta"]["intermediate_conclusions"].values())
    return facts

# ==================================================================================================

# Model loading
device = "cuda"
tokenizer = transformers.AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
model = transformers.AutoModelForCausalLM.from_pretrained(
    "NousResearch/Llama-2-7b-hf",
    quantization_config=transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    ),
    torch_dtype=torch.bfloat16,
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
peft_config = LoraConfig(
    r=128,
    lora_alpha=16,
    target_modules=find_all_linear_names(model),
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
print_trainable_parameters(model)
print("[-] Model loaded")

chkp_path = "checkpoints/implication_20240517-135825_Llama-2-7b-hf.pth.tar"
checkpoint = torch.load(chkp_path, map_location="cpu")
model.load_state_dict(checkpoint['state_dict'])
print(f"[-] Model checkpoint loaded: {chkp_path}")

with open("dataset.json") as f:
    data = json.load(f)
    X = [s["x"] for s in data]
    Y = [s["y"] for s in data]

# Querying
print("[-] Querying the model...")
prompt = "You can answer only with \"true\" or \"false\". Is the following statement true? Statement: {s}. Answer: "
y_hat = []
for batch in tqdm(list(make_batch(X, 8))):
    batch = [prompt.format(s=b) for b in batch]
    answers = get_beliefs(model, tokenizer, batch)
    for l in list(zip(batch, [a.split("Answer:")[1].lower() for a in answers])): print(l)
    answers = [int("true" in a.split("Answer:")[1].lower()) for a in answers]
    y_hat += answers

print(f"# trues: {len([k for k in y_hat if k == 1])}/{len(y_hat)}")
print(f"Score: {f1_score(Y, y_hat)}")


