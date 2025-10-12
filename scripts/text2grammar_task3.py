import pandas as pd 
import os
import pickle
from datetime import datetime, timezone
from openai import OpenAI  # pip install openai
import nltk
from nltk.tokenize import sent_tokenize
import os
from pprint import pprint
import pandas as pd 
from mt_reasoning.utils import prompts_util, clients_util 
from tqdm import tqdm
import importlib
from dotenv import load_dotenv
import random
import string
import argparse

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument('--model_vllm', type=str, default=os.environ.get("MODEL_VLLM", "/home/snt/projects_lujun/base_models/gemma-2-2b-it"), help='Path to the VLLM model')
parser.add_argument('--port', type=str, default=os.environ.get("VLLM_PORT", "1997"), help='Port for VLLM service')
parser.add_argument('--sentence_list_size', type=int, default=2, help='Input Sentence Num to Concatenate')
parser.add_argument('--grammar_size', type=int, default=5, help='Input Grammar Descriptions Num to Evaluate')
parser.add_argument('--test_length', type=int, default=1000, help='The Tests Num We Want to evaluate')

args = parser.parse_args()

model_vllm = args.model_vllm
PORT = args.port
sentence_list_size = args.sentence_list_size
grammar_size = args.grammar_size
test_length = args.test_length



source_df = pd.read_json("data/extraction_pdf/datasets/df_samples.jsonl", lines=True)

## uv run vllm serve /home/snt/projects_lujun/base_models/gemma-2-2b-it --host 0.0.0.0 --port 1997 --max-model-len 2048 --max-num-seqs 2 --gpu-memory-utilization 0.7



importlib.reload(prompts_util)
importlib.reload(clients_util)

nltk.download('punkt')

## Open AI Settings
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "0.5"))


## VllM settings
# model_vllm = os.environ.get("MODEL_VLLM", "/home/snt/projects_lujun/base_models/gemma-2-2b-it")
IP = os.environ.get("VLLM_IP", "0.0.0.0")
# PORT = os.environ.get("VLLM_PORT", "1997")
server_url = f"http://{IP}:{PORT}/v1"
print (server_url)

if "gpt" in model_vllm:
    vllm_client = OpenAI(api_key=OPENAI_API_KEY)
    vllm_extra={"logprobs": False, "top_logprobs": grammar_size}
elif "thinking" in model_vllm.lower():
    vllm_client = OpenAI(base_url=server_url)
    vllm_extra={"logprobs": False, "chat_template_kwargs": {"thinking": True}}
else:
    vllm_client = OpenAI(base_url=server_url)
    vllm_extra={"logprobs": False}
## Experimental Settings
# sentence_list_size = 2  # Input Sentence Num to Concatenate
# grammar_size = 5  # Input Grammar Descriptions Num to Evaluate
# test_length = 1000  # The Tests Num We Want to evaluate

letters = list(string.ascii_uppercase)  # ['A', 'B', 'C', ..., 'Z']

time_now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

project_dir = os.environ.get("PROJECT_DIR", None)
output_dir = os.path.join(project_dir, "data/extraction_pdf/datasets")

import glob
pattern = os.path.join(
    output_dir,
    f"task3_*_{grammar_size}_{sentence_list_size}_{test_length}_{model_vllm.split('/')[-1]}.jsonl"
)

matches = glob.glob(pattern)
if matches:
    matches.sort(key=os.path.getmtime, reverse=True)
    output_path = matches[0]
    print(f"Resuming from existing file: {output_path}")
    finished_lines = len(pd.read_json(output_path, lines=True))
    print(f"Already processed {finished_lines} lines.")
else:
    finished_lines = 0
    output_path = os.path.join(output_dir,  f"task3_{time_now}_{grammar_size}_{sentence_list_size}_{test_length}_{model_vllm.split('/')[-1]}.jsonl")



for idx in tqdm(range(test_length), desc="Processing"):
    if idx < finished_lines:
        continue  # Skip already processed rows

    ## Randomly Sample a Row from the Source DataFrame
    rows = (
        source_df[source_df['grammar_points_descriptions'].isin(
            random.sample(list(source_df['grammar_points_descriptions'].unique()), sentence_list_size)
        )]
        .groupby('grammar_points_descriptions', group_keys=False)
        .sample(1)
        .reset_index(drop=True)
    )


    sentence_list = rows["luxembourg"].tolist()
    grammar_points = rows["grammar_points_descriptions"].drop_duplicates().tolist()
    assert len(sentence_list) == len(grammar_points), "Grammar Sentence mismatch."

    random.shuffle(sentence_list)
    paragraph = " ".join(sentence_list)
    num_grammars = sentence_list_size

    ## Grammar List Generation
    k = max(0, grammar_size - sentence_list_size)
    available = source_df.loc[~source_df['grammar_points_descriptions'].isin(grammar_points), 'grammar_points_descriptions'].drop_duplicates()
    opposite_source_sentence_list = (available.sample(n=k, replace=(k>len(available))).tolist() if k>0 and not available.empty else [])

    full_grammar_list = grammar_points + opposite_source_sentence_list
    random.shuffle(full_grammar_list)
    grammars_index = [full_grammar_list.index(g) if g in full_grammar_list else -1 for g in grammar_points]

    assert len(grammars_index) == sentence_list_size, "Grammar not found in the list."
    assert len(full_grammar_list) == grammar_size, "Sentence list size mismatch."

    option_labels = letters[:grammar_size]
    correct_grammars_letter = [option_labels[i] for i in grammars_index]
    labeled_grammars_list = [
        f"{label}. {desc}" for label, desc in zip(option_labels, full_grammar_list)
    ]
    input_dict = {
        "NUM_GRAMMARS": num_grammars,
        "PARAGRAPH":  paragraph,
        "GRAMMAR_LIST": "\n".join(labeled_grammars_list),
    }

    output_dict, input_prompt, log_probs = clients_util.generate_with_calling_api(
        client=vllm_client,
        system_prompt_template_path="prompts/system/system_prompt_translation.jinja",
        input_prompt_template_path="prompts/evaluation/prompt_sentence_classification_task_3.jinja",  # Use simple, complecated one confuse the models
        input_text_dict=input_dict,
        model=model_vllm,
        vllm_extra=vllm_extra
    )
    
    row = {
        "sentence_list": sentence_list,
        "grammar_points": grammar_points,
        "full_grammar_list": full_grammar_list,
        "option_labels": option_labels,
        "labeled_grammars_list": labeled_grammars_list,
        "input_dict": input_dict,
        "input_prompt": input_prompt,
        "log_probs": log_probs,
        "task3_dict": output_dict,
        "correct_grammars_letter": correct_grammars_letter,
    }
    
    updated_row = pd.DataFrame([row])
    if not os.path.exists(output_dir):  
        os.makedirs(output_dir)
    if not os.path.exists(output_path):
        updated_row.to_json(output_path, orient="records", lines=True)
    else:
        updated_row.to_json(output_path, orient="records", lines=True, mode="a")
    
    # print(output_dict)
    # print("----------------------------------------------")
    # pprint(output_dict, indent=2, width=150, sort_dicts=False)