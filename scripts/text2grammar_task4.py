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

args = parser.parse_args()

model_vllm = args.model_vllm
PORT = args.port



source_df = pd.read_json("data/extraction_pdf/datasets/df_samples.jsonl", lines=True)

## uv run vllm serve /home/snt/projects_lujun/base_models/gemma-2-2b-it --host 0.0.0.0 --port 1997 --max-model-len 2048 --max-num-seqs 2 --gpu-memory-utilization 0.7

adverserial_df = pd.read_json("data/extraction_pdf/datasets/back_checking_20250930_170933.jsonl", lines=True)

for i in range(len(source_df)):
    source_lux = source_df.loc[i, "luxembourg"]
    adverserial_lux = adverserial_df.loc[i, "luxembourg"]

    assert source_lux == adverserial_lux, f"mismatch at {i}"
    source_df.loc[i, "adverserial"] = adverserial_df.loc[i, "back_checking_dict"]["false_sentence"]


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
letters = list(string.ascii_uppercase)  # ['A', 'B', 'C', ..., 'Z']

vllm_extra={"logprobs": True, "top_logprobs": 2}
time_now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
project_dir = os.environ.get("PROJECT_DIR", None)
output_dir = os.path.join(project_dir, "data/extraction_pdf/datasets")

import glob
pattern = os.path.join(
    output_dir,
    f"task4_*_{model_vllm.split('/')[-1]}.jsonl"
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
    output_path = os.path.join(output_dir, f"task4_{time_now}_{model_vllm.split('/')[-1]}.jsonl")


for index, row in tqdm(source_df.iterrows(), total=len(source_df)):
    if index < finished_lines:
        continue  # Skip already processed rows
    grammar_desc = row['grammar_points_descriptions']
    lux_sentence_correct = row['luxembourg']
    lux_sentence_adverserial = row['adverserial']
    lux_sentence_list = [lux_sentence_correct, lux_sentence_adverserial]
    random.shuffle(lux_sentence_list)
    assert lux_sentence_correct in lux_sentence_list, "Correct sentence not in the list."
    correct_sentence_index = lux_sentence_list.index(lux_sentence_correct)
    assert correct_sentence_index != -1, "Correct sentence index not found."
    option_labels = letters[:2]
    correct_sentence_letter = option_labels[correct_sentence_index]

    ## mp
    input_dict_mp = {
        "FIRST_LUXEMBOURGISH_SENTENCE": lux_sentence_list[0],
        "SECOND_LUXEMBOURGISH_SENTENCE": lux_sentence_list[1],
    }

    # mp_single 1
    input_dict_mp_single_1 = {
        "LUXEMBOURGISH_SENTENCE": lux_sentence_list[0],
    }
    row["mp_single_1_ground_truth"] = "Yes" if correct_sentence_index == 0 else "No"

    # mp_single 2
    input_dict_mp_single_2 = {
        "LUXEMBOURGISH_SENTENCE": lux_sentence_list[1],
    }
    row["mp_single_2_ground_truth"] = "Yes" if correct_sentence_index == 1 else "No"

    # mp_with_grammar desc
    input_dict_mp_with_grammar = {
        "GRAMMAR_DESCRIPTION": grammar_desc,
        "FIRST_LUXEMBOURGISH_SENTENCE": lux_sentence_list[0],
        "SECOND_LUXEMBOURGISH_SENTENCE": lux_sentence_list[1],
    }

    row["correct_sentence_letter"] = correct_sentence_letter

    output_dict_mp, input_prompt_mp, log_prob_mp = clients_util.generate_with_calling_api(
        client=vllm_client,
        system_prompt_template_path="prompts/system/system_prompt_translation.jinja",
        input_prompt_template_path="prompts/evaluation/prompt_grammar_classification_task_4_mp.jinja",  # Use simple, complecated one confuse the models
        input_text_dict=input_dict_mp,
        model=model_vllm,
        vllm_extra=vllm_extra,
    )

    output_dict_mp_single_1, input_prompt_mp_single_1, log_prob_mp_single_1 = clients_util.generate_with_calling_api(
        client=vllm_client,
        system_prompt_template_path="prompts/system/system_prompt_translation.jinja",
        input_prompt_template_path="prompts/evaluation/prompt_grammar_classification_task_4_mp_single.jinja",  # Use simple, complecated one confuse the models
        input_text_dict=input_dict_mp_single_1,
        model=model_vllm,
        vllm_extra=vllm_extra,
    )

    output_dict_mp_single_2, input_prompt_mp_single_2, log_prob_mp_single_2 = clients_util.generate_with_calling_api(
        client=vllm_client,
        system_prompt_template_path="prompts/system/system_prompt_translation.jinja",
        input_prompt_template_path="prompts/evaluation/prompt_grammar_classification_task_4_mp_single.jinja",  # Use simple, complecated one confuse the models
        input_text_dict=input_dict_mp_single_2,
        model=model_vllm,
        vllm_extra=vllm_extra,
    )

    output_dict_mp_with_grammar_desc, input_prompt_mp_with_grammar_desc, log_prob_mp_with_grammar_desc = clients_util.generate_with_calling_api(
        client=vllm_client,
        system_prompt_template_path="prompts/system/system_prompt_translation.jinja",
        input_prompt_template_path="prompts/evaluation/prompt_grammar_classification_task_4_mp_with_grammar_desc.jinja",  # Use simple, complecated one confuse the models
        input_text_dict=input_dict_mp_with_grammar,
        model=model_vllm,
        vllm_extra=vllm_extra,
    )
    

    row["input_prompt_mp"] = input_prompt_mp
    row["output_dict_mp"] = output_dict_mp
    row["log_prob_mp"] = log_prob_mp
    row["output_dict_mp_single_1"] = output_dict_mp_single_1
    row["input_prompt_mp_single_1"] = input_prompt_mp_single_1
    row["log_prob_mp_single_1"] = log_prob_mp_single_1
    row["output_dict_mp_single_2"] = output_dict_mp_single_2
    row["input_prompt_mp_single_2"] = input_prompt_mp_single_2
    row["log_prob_mp_single_2"] = log_prob_mp_single_2
    row["output_dict_mp_with_grammar_desc"] = output_dict_mp_with_grammar_desc
    row["input_prompt_mp_with_grammar_desc"] = input_prompt_mp_with_grammar_desc
    row["log_prob_mp_with_grammar_desc"] = log_prob_mp_with_grammar_desc


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