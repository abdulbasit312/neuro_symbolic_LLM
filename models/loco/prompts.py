import regex
import random

FALSE_LABELS = [
    "no",
    "false",
    "negative"
]
TRUE_LABELS = [
    "yes",
    "true",
    "positive"
]

#FALSETRUE = ["false", "true"]
#INSTRUCTION_PROMPT = "You can answer only with \"true\" or \"false\". Is the fact true? "

DEFAULT_PROMPT_FORMAT = 0
FALSETRUE = ["no", "yes"]
INSTRUCTION_PROMPT = "Tell if a fact is true or false. You can answer only with \"yes\" or \"no\". "

# 0-3   zero shot
# 4-7   few shot
FORMATS = [
    {
        "prompt": "You can answer only with \"true\" or \"false\". Is the fact true? Fact: {fact} Answer: ",
        "label": "true",
        "negative": "false"
    },
    {
        "prompt": "You can answer only with \"yes\" or \"no\". Is the fact true? Fact: {fact} Answer: ",
        "label": "yes",
        "negative": "no"
    },
    {
        "prompt": "Answer only with \"correct\" or \"incorrect\". Is the statement accurate? Statement: {fact} Answer: ",
        "label": "correct",
        "negative": "incorrect"
    },
    {
        "prompt": "Respond only with \"right\" or \"wrong\". Does the claim hold true? Claim: {fact} Answer: ",
        "label": "right",
        "negative": "wrong"
    },
    {
        "prompt": "Fact: the earth is round. Label: true.\nFact: the sun is cold. Label: false.\nFact: {fact} Label:",
        "label": "true",
        "negative": "false"
    },
    {
        "prompt": "Fact: the earth is round. Label: yes.\nFact: the sun is cold. Label: no.\nFact: {fact} Label:",
        "label": "yes",
        "negative": "no"
    },
    {
        "prompt": "Statement: the earth is round. Label: correct.\nStatement: the sun is cold. Label: incorrect.\nStatement: {fact} Label:",
        "label": "correct",
        "negative": "incorrect"
    },
    {
        "prompt": "Claim: the earth is round. Label: right.\nClaim: the sun is cold. Label: wrong.\nClaim: {fact} Label:",
        "label": "right",
        "negative": "wrong"
    }

]

def preprocess_batch(batch, prompt_format):
    """ Formats input facts with the structured prompt with instructions """
    formatted_batch = [FORMATS[prompt_format]["prompt"].format(fact=f) for f in batch]
    return formatted_batch

def postprocess_answers(prompts, answers, prompt_format):
    """ Convert decoder-only output (prompt+answer) to the binary truth assignment given a prompt format """
    answ_without_prefix = [s.replace(prompts[idx], "") for idx, s in enumerate(answers)]
    labels = [int(FORMATS[prompt_format]["label"].lower() in answ.lower()) for answ in answ_without_prefix]
    return answ_without_prefix, labels

def prompt_answer(type:str, text:str) -> str:
    if type == "decoder":
        return text
    else:
        return f"$answer$ = {text} ;"

# ==========================================================

def in_f_flant5(questions):
    return [
        f'{q} Answer with "{YES}" or "{NO}": '
        for q in questions
    ]        

def in_f_macaw(questions):
    return [
        f'$answer$ ; $mcoptions$ = (A) {YES}; $question$ = ' + q
        for q in questions
    ]

def out_f_macaw(answers):
    return [(re.split("=|;", a)[1]).strip() for a in answers]

    
