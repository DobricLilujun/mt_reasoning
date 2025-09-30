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
load_dotenv()

importlib.reload(prompts_util)
importlib.reload(clients_util)

nltk.download('punkt')


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "1.0"))
model_extraction = os.environ.get("MODEL_EXTRACTION", "gpt-5")
openai_client = OpenAI(api_key=OPENAI_API_KEY)
df = pd.read_json('data/extraction_pdf/datasets/df_samples.jsonl', lines=True)

time_now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
output_dir = "/home/snt/projects_lujun/mt_reasoning/data/extraction_pdf/datasets"
output_path = os.path.join(output_dir, f"back_checking_{time_now}.jsonl")

for index, row in tqdm(df.iterrows(), total=len(df)):
    input_dict = {
        "LUXEMBOURGISH_SENTENCE": row['luxembourg'],
        "ENGLISH_SENTENCE": row['english'],
        "GRAMMAR_DESCRIPTION": row['grammar_points_descriptions'],
    }

    output_dict, input_prompt = clients_util.generate_with_calling_openai_api(
        client=openai_client,
        system_prompt_template_path="prompts/system/system_prompt_translation.jinja",
        input_prompt_template_path="prompts/extraction/prompt_back_checking_with_grammar.jinja",  # Use simple, complecated one confuse the models
        input_text_dict=input_dict,
        model=model_extraction,
    )
    row["input_prompt"] = input_prompt
    row["back_checking_dict"] = output_dict
    updated_row = pd.DataFrame([row])
    if not os.path.exists(output_dir):  
        os.makedirs(output_dir)
    if not os.path.exists(output_path):
        updated_row.to_json(output_path, orient="records", lines=True)
    else:
        updated_row.to_json(output_path, orient="records", lines=True, mode="a")
    
    print(output_dict)
    print("----------------------------------------------")
    pprint(output_dict, indent=2, width=150, sort_dicts=False)