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
parser.add_argument('--sentence_list_size', type=int, default=3, help='Size of the sentence list')

args = parser.parse_args()

model_vllm = args.model_vllm
PORT = args.port
sentence_list_size = args.sentence_list_size


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
vllm_client = OpenAI(base_url=server_url)

## Experimental Settings
# sentence_list_size = 3
letters = list(string.ascii_uppercase)  # ['A', 'B', 'C', ..., 'Z']
vllm_extra={"logprobs": True, "top_logprobs": sentence_list_size}
time_now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
project_dir = os.environ.get("PROJECT_DIR", None)
output_dir = os.path.join(project_dir, "data/extraction_pdf/datasets")


import glob
pattern = os.path.join(
    output_dir,
    f"task2_*_{sentence_list_size}_{model_vllm.split('/')[-1]}.jsonl"
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
    output_path = os.path.join(output_dir, f"task2_{time_now}_{sentence_list_size}_{model_vllm.split('/')[-1]}.jsonl")


grammar_df = source_df.drop_duplicates(subset=['grammar_points_descriptions'])

print (f"Unique grammar descriptions: {len(grammar_df)}")

for index, row in tqdm(grammar_df.iterrows(), total=len(grammar_df)):
    if index < finished_lines:
        continue  # Skip already processed rows
    grammar_desc = row['grammar_points_descriptions']
    sentence_lux = row['luxembourg']
    opposite_source_sentence_list = grammar_df[grammar_df['grammar_points_descriptions'] != grammar_desc]['luxembourg'].sample(sentence_list_size-1).tolist()
    full_list = [sentence_lux] + opposite_source_sentence_list
    random.shuffle(full_list)
    sentence_index = full_list.index(sentence_lux)

    assert sentence_index != -1, "Sentence not found in the list."
    assert len(full_list) == sentence_list_size, "Sentence list size mismatch."

    option_labels = letters[:sentence_list_size]
    correct_sentence_letter = option_labels[sentence_index]
    labeled_sentence_list = [
        f"{label}. {desc}" for label, desc in zip(option_labels, full_list)
    ]
    input_dict = {
        "GRAMMAR_POINT": grammar_desc,
        "LIST_OF_SENTENCES": "\n".join(labeled_sentence_list),
    }


    output_dict, input_prompt, log_probs = clients_util.generate_with_calling_api(
        client=vllm_client,
        system_prompt_template_path="prompts/system/system_prompt_translation.jinja",
        input_prompt_template_path="prompts/evaluation/prompt_sentence_classification_task_2.jinja",  # Use simple, complecated one confuse the models
        input_text_dict=input_dict,
        model=model_vllm,
        vllm_extra=vllm_extra
    )
    
    row["input_prompt"] = input_prompt
    row["log_probs"] = log_probs
    row["task2_dict"] = output_dict
    row["correct_sentence_letter"] = correct_sentence_letter
    
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

    