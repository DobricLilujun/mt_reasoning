import os
import pickle
from dotenv import load_dotenv
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
load_dotenv()

importlib.reload(prompts_util)
importlib.reload(clients_util)

nltk.download('punkt')


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
print (f"OPENAI_API_KEY: {OPENAI_API_KEY}")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini") 
TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "1.0"))
model_extraction = os.environ.get("MODEL_EXTRACTION", "gpt-5-mini")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

with open("data/extraction_pdf/outputs/grammer_info.pkl", "rb") as f:
    book_extraction_list = pickle.load(f)

chapter_category_map = {
    "1. Introduction": None,
    "2. Sketch of the Sociohistorical and Sociolinguistic": None,
    "3. Phonetics and Phonology": None,
    "4. Morphosyntax": ["Morphology", "Syntax"],
    "5. Selected Syntactic Characteristics": ["Syntax"],
    "6. Lexical Structures": ["Morphology"],
    "7. Language Variation and Change": ["Irregular_form"],
    "8. Conclusion": None
}
time_now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
output_dir = "/home/snt/projects_lujun/mt_reasoning/data/extraction_pdf/extraction_table_texts"
output_path = os.path.join(output_dir, f"extraction_{time_now}.jsonl")


def segment_by_fixed_size_with_step(text, group_size=3, step=2, language='english', keep_tail=False):
    sents = sent_tokenize(text, language=language)
    segments, idx_spans = [], []
    i = 0
    n = len(sents)
    while i < n:
        j = i + group_size
        if j <= n:
            seg = " ".join(sents[i:j])
            segments.append(seg)
            idx_spans.append((i, j))
        elif keep_tail and i < n:
            seg = " ".join(sents[i:n])
            segments.append(seg)
            idx_spans.append((i, n))
            break
        else:
            break
        i += step
    return segments, idx_spans, sents


# Extract from the table row by row
def nearest_text_neighbors(content_dict_list, i):
    n = len(content_dict_list)
    prev_idx = next((i-k for k in range(1, i+1)
                     if content_dict_list[i-k].get("dtype") == "text"), None)
    next_idx = next((i+k for k in range(1, n-i)
                     if content_dict_list[i+k].get("dtype") == "text"), None)
    content_dict_prev = content_dict_list[prev_idx] if prev_idx is not None else None
    content_dict_next = content_dict_list[next_idx] if next_idx is not None else None
    return content_dict_prev, content_dict_next

## Extraction with text Only
for grammer_section in book_extraction_list:
    chapter = grammer_section['section_title']
    print(f"Processing chapter: {chapter}")
    if chapter in chapter_category_map and chapter_category_map[chapter]:
        content_dict_list = grammer_section["content_dict_list"]
        for i, content_dict in tqdm(enumerate(content_dict_list), total=len(content_dict_list), desc="Processing", unit="item"):

            if content_dict["dtype"] == "text":
                # continue
                text_input = content_dict["text"]
                segments, spans, sents = segment_by_fixed_size_with_step(text_input, group_size=6, step=3, language='english', keep_tail=False)
                for m, (seg, (a, b)) in enumerate(tqdm(zip(segments, spans), total=min(len(segments), len(spans)), desc="Processing segments in texts", unit="pair"), 1):

                    # Update the input_dict with the current segment
                    input_dict = {
                        "text": seg,
                        "min_words_per_sentence": 20,
                        "target_min_words": 22,
                        "target_max_words": 28
                    }
                    # Give template, give the dict as input, get the final output dict
                    output_dict, input_prompt = clients_util.generate_with_calling_openai_api(
                        client=openai_client,
                        system_prompt_template_path="prompts/system/system_prompt_translation.jinja",
                        input_prompt_template_path="prompts/extraction/prompt_extraction_with_grammer_text.jinja",
                        input_text_dict=input_dict,
                        model=model_extraction,
                    )
                    row = {
                        "chapter": chapter,
                        "category": chapter_category_map[chapter],
                        "content_dict": content_dict,        #
                        "index_in_content_dict_list": i,
                        "dtype": "text",
                        "row_index": 0,
                        "input_prompt": input_prompt,
                        "output_dict": output_dict,        
                        "input_dict": input_dict            
                    }
                    updated_row = pd.DataFrame([row])
                    if not os.path.exists(output_dir):  
                        os.makedirs(output_dir)
                    if not os.path.exists(output_path):
                        updated_row.to_json(output_path, orient="records", lines=True)
                    else:
                        updated_row.to_json(output_path, orient="records", lines=True, mode="a")
                    
            elif content_dict["dtype"] == "table":

                content_dict_prev, content_dict_next = nearest_text_neighbors(content_dict_list, i)
                surrounding_context = content_dict_prev["text"] + "\n" + content_dict_next["text"] if content_dict_prev and content_dict_next else ""
                table_df  = content_dict["df"]
                full_table_html = content_dict["html"]
                # first_two_rows_string = str(table_df.iloc[0].to_dict()) + "\n" + str(table_df.iloc[1].to_dict())
                for row_index, row in table_df.iterrows():
                    if row_index <= 1:
                        continue
                    print("Processing row:", row_index)
                    specific_row = row.to_dict()
                    grammar_table = full_table_html
                    input_dict = {
                        "GRAMMAR_TABLE": grammar_table,
                        "SPECIFIC_ROW": specific_row,
                        "min_words_per_sentence": 20,
                        "target_min_words": 22,
                        "target_max_words": 28
                    }
                    output_dict, input_prompt = clients_util.generate_with_calling_openai_api(
                        client=openai_client,
                        system_prompt_template_path="prompts/system/system_prompt_translation.jinja",
                        input_prompt_template_path="prompts/extraction/prompt_extraction_with_grammer_table_simple.jinja",  # Use simple, complecated one confuse the models
                        input_text_dict=input_dict,
                        model=model_extraction,
                    )

                    row = {
                        "chapter": chapter,
                        "category": chapter_category_map[chapter],
                        "content_dict": content_dict,        #
                        "index_in_content_dict_list": i,
                        "dtype": "table",
                        "row_index": row_index,
                        "input_prompt": input_prompt,
                        "output_dict": output_dict,        
                        "input_dict": input_dict            
                    }
                    updated_row = pd.DataFrame([row])

                    if not os.path.exists(output_dir):  
                        os.makedirs(output_dir)
                    if not os.path.exists(output_path):
                        updated_row.to_json(output_path, orient="records", lines=True)
                    else:
                        updated_row.to_json(output_path, orient="records", lines=True, mode="a")
                    
            else:
                print("Unknown dtype:", content_dict["dtype"])
                raise ValueError("Unknown dtype")

