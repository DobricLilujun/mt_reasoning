# uv run vllm serve /home/snt/projects_lujun/base_models/Qwen3-4B-Thinking-2507 --gpu_memory_utilization 0.5 --max_model_len 32768 --port 8000 --host 0.0.0.0

# uv run vllm serve /home/snt/projects_lujun/base_models/DeepSeek-R1-Distill-Llama-8B --enable-reasoning --reasoning-parser deepseek_r1 --gpu_memory_utilization 0.5 --max_model_len 32768 --port 8000 --host 0.0.0.0 

import pandas as pd
import requests
from openai import OpenAI
import importlib
from datetime import datetime
from mt_reasoning.util import prompts_util, clients_util 
from tqdm import tqdm

importlib.reload(prompts_util)
importlib.reload(clients_util)

import os
ts = datetime.now().strftime("%m%d%H%M")  

#### Setttings

df = pd.read_json("data/source/flore200.jsonl", lines=True)
openai_api_key = "sk-pro"
openai_api_base = "http://0.0.0.0:8000/v1/chat/completions"
openai_api_base = "https://api.openai.com/v1/chat/completions"
# model_path = "/home/snt/projects_lujun/base_models/Qwen3-4B-Thinking-2507"
model_path = "/home/snt/projects_lujun/base_models/DeepSeek-R1-Distill-Llama-8B"
model_name = model_path.split("/")[-1]
# model_path = "o3-mini-2025-01-31"
output_path_folder = "data/outputs/flore200_eval"
iso639_df = prompts_util.load_iso639_df()
tgt_langs = [ "Assamese", "Luxembourgish", "Maltese", "Javanese", "Lingala", "Hindi", "German", "Modern Standard Arabic", "Standard Malay","Swahili"]
tgt_codes = [ "asm", "ltz", "mlt", "jav", "lin", "hin", "ger", "ara", "may", "swa"]


client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
eng_code = "eng"
all_langs_code = df['iso_639_3'].dropna().unique()
df_eng = df[df['iso_639_3'] == eng_code]

for tgt_code in tgt_codes:
    print(f"Translating {eng_code} <-> {tgt_code}")

    ### English To LRLs
    out_path = f"{output_path_folder}/results_{eng_code}_{tgt_code}_{model_name}.jsonl"
    if os.path.exists(out_path):
        try:
            df_prev = pd.read_json(out_path, lines=True)
            idx_prev = len(df_prev)
        except Exception:
            idx_prev = 0
    else:
        idx_prev = 0

    for idx, row in tqdm(df_eng.iterrows(), total=df_eng.shape[0], desc="Eng rows", unit="row"):
        if idx < idx_prev:
            continue
        update_row = row.copy()
        translation_prompt = prompts_util.translate_prompt(row['text'], eng_code, tgt_code)
        reasoning_started_at = datetime.now()
        translation, translation_reasoning_path = clients_util.generate_with_calling_vllm(openai_api_base, model_path, translation_prompt, api_key=openai_api_key)
        reasoning_ended_at = datetime.now()
        reasoning_elapsed_sec = (reasoning_ended_at - reasoning_started_at).total_seconds()
        update_row['translation'] = translation
        update_row['translation_reasoning_path'] = str(translation_reasoning_path)
        update_row['reasoning_started_at'] = reasoning_started_at.isoformat()
        update_row['reasoning_ended_at'] = reasoning_ended_at.isoformat()
        update_row['reasoning_elapsed_sec'] = reasoning_elapsed_sec
        updated_df = pd.DataFrame([update_row])
        mode = "w" if idx_prev == 0 else "a"
        updated_df.to_json(out_path, orient="records", lines=True, mode=mode)

    ### LRLs To English
    out_path = f"{output_path_folder}/results_{tgt_code}_{eng_code}_{model_name}.jsonl"
    if os.path.exists(out_path):
        try:
            df_prev = pd.read_json(out_path, lines=True)
            idx_prev = len(df_prev)
        except Exception:
            idx_prev = 0
    else:
        idx_prev = 0

    for idx, row in tqdm(df_eng.iterrows(), total=df_eng.shape[0], desc="Eng rows", unit="row"):
        if idx < idx_prev:
            continue
        update_row = row.copy()
        translation_prompt = prompts_util.translate_prompt(row['text'], tgt_code, eng_code)
        reasoning_started_at = datetime.now()
        translation, translation_reasoning_path = clients_util.generate_with_calling_vllm(openai_api_base, model_path, translation_prompt, api_key=openai_api_key)
        reasoning_ended_at = datetime.now()
        reasoning_elapsed_sec = (reasoning_ended_at - reasoning_started_at).total_seconds()
        update_row['translation'] = translation
        update_row['translation_reasoning_path'] = str(translation_reasoning_path)
        update_row['reasoning_started_at'] = reasoning_started_at.isoformat()
        update_row['reasoning_ended_at'] = reasoning_ended_at.isoformat()
        update_row['reasoning_elapsed_sec'] = reasoning_elapsed_sec
        updated_df = pd.DataFrame([update_row])
        mode = "w" if idx_prev == 0 else "a"
        updated_df.to_json(out_path, orient="records", lines=True, mode=mode)