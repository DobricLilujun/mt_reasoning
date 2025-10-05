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
import sacrebleu 
from mt_reasoning.utils import prompts_util, clients_util, eval_util
from tqdm import tqdm
import importlib
from dotenv import load_dotenv
import random
import string
import argparse
from comet import download_model, load_from_checkpoint
from typing import List, Dict
from huggingface_hub import snapshot_download, login

load_dotenv()



parser = argparse.ArgumentParser()
parser.add_argument('--model_vllm', type=str, default=os.environ.get("MODEL_VLLM", "/home/snt/projects_lujun/base_models/gemma-2-2b-it"), help='Path to the VLLM model')
parser.add_argument('--port', type=str, default=os.environ.get("VLLM_PORT", "1997"), help='Port for VLLM service')

args = parser.parse_args()

model_vllm = args.model_vllm
PORT = args.port

# model_vllm = "/home/snt/projects_lujun/base_models/gemma-2-2b-it"
# PORT = 1997

source_df = pd.read_json("data/performance_eval/eng_lux_concat.jsonl", lines=True)

## uv run vllm serve /home/snt/projects_lujun/base_models/gemma-2-2b-it --host 0.0.0.0 --port 1997 --max-model-len 2048 --max-num-seqs 2 --gpu-memory-utilization 0.7


importlib.reload(prompts_util)
importlib.reload(clients_util)

nltk.download('punkt')

## Open AI Settings
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "0.5"))
HUGGINGFACE_TOKEN= os.environ.get("HUGGINGFACE_TOKEN", "")
login(token=HUGGINGFACE_TOKEN)
## VllM settings
# model_vllm = os.environ.get("MODEL_VLLM", "/home/snt/projects_lujun/base_models/gemma-2-2b-it")
IP = os.environ.get("VLLM_IP", "0.0.0.0")
# PORT = os.environ.get("VLLM_PORT", "1997")
server_url = f"http://{IP}:{PORT}/v1"
print (server_url)

if "gpt" in model_vllm:
    vllm_client = OpenAI(api_key=OPENAI_API_KEY)
    vllm_extra={"logprobs": False}
else:
    vllm_client = OpenAI(base_url=server_url)
    vllm_extra={"logprobs": False}

## Experimental Settings
# grammar_list_size = 5
letters = list(string.ascii_uppercase)  # ['A', 'B', 'C', ..., 'Z']

# vllm_extra={"logprobs": True, "top_logprobs": grammar_list_size}
time_now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

project_dir = os.environ.get("PROJECT_DIR", None)
output_dir = os.path.join(project_dir, "data/performance_eval")

import glob
pattern = os.path.join(
    output_dir,
    f"task_eval_translation_*_{model_vllm.split('/')[-1]}.jsonl"
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
    output_path = os.path.join(output_dir, f"task_eval_translation_{time_now}_{model_vllm.split('/')[-1]}.jsonl")


print(f"Starting evaluation for model: {model_vllm}")
comet_model = eval_util.init_comet_model(model_name="Unbabel/wmt23-cometkiwi-da-xl", gpus=1)
comet_model_small = eval_util.init_comet_model(model_name="Unbabel/wmt22-cometkiwi-da", gpus=1)
chrf_metric = sacrebleu.CHRF(word_order=3)

for index, row in tqdm(source_df.iterrows(), total=len(source_df)):
    if index < finished_lines:
        continue  # Skip already processed rows

    lux_sentence = row['text_ltz']
    eng_sentence = row['text_eng']

    input_dict = {
        "ENGLISH_SENTENCE": eng_sentence,
    }

    output_dict, input_prompt, log_probs = clients_util.generate_with_calling_api(
        client=vllm_client,
        system_prompt_template_path="prompts/system/system_prompt_translation.jinja",
        input_prompt_template_path="prompts/evaluation/prompt_translation_eng_lux.jinja",  # Use simple, complecated one confuse the models
        input_text_dict=input_dict,
        model=model_vllm,
        vllm_extra=vllm_extra
    )
    
    row["input_prompt"] = input_prompt
    row["task_translation_dict"] = output_dict
    translated_text = output_dict.get("translation", "").strip()

    # Cometkiwi
    comet_score = eval_util.evaluate_with_comet_ref(model=comet_model, src=[eng_sentence], mt=[translated_text], ref=[lux_sentence])["system_score"]
    comet_score_small = eval_util.evaluate_with_comet_ref(model=comet_model_small, src=[eng_sentence], mt=[translated_text], ref=[lux_sentence])["system_score"]

    # spbleu score
    spbleu_score = sacrebleu.corpus_bleu([translated_text], [[lux_sentence]], tokenize="flores200").score

    # Chrf++ score
    charf_score = chrf_metric.sentence_score(translated_text, [lux_sentence]).score

    row["CometScore"] = comet_score
    row["CometScoreSmall"] = comet_score_small
    row["spbleu_score"] = spbleu_score
    row["chrf_score"] = charf_score

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