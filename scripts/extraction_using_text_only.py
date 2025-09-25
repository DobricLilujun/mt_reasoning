import os
import pickle
from openai import OpenAI  # pip install openai
import nltk
from nltk.tokenize import sent_tokenize
import os
from pprint import pprint

from mt_reasoning.utils import prompts_util, clients_util 
from tqdm import tqdm
import importlib

importlib.reload(prompts_util)
importlib.reload(clients_util)

nltk.download('punkt')

input_dict = {
    "text": "Your input text here",
    "min_words_per_sentence": 20,
    "target_min_words": 22,
    "target_max_words": 28

}
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini") 
TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "1.0"))

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


## Extraction with text Only
for grammer_section in book_extraction_list:
    chapter = grammer_section['section_title']
    print(chapter)
    if chapter in chapter_category_map and chapter_category_map[chapter] and chapter == "5. Selected Syntactic Characteristics":
        content_dict_list = grammer_section["content_dict_list"]
        for content_dict in content_dict_list:
            if content_dict["dtype"] == "text":
                test_input = content_dict["text"]
                segments, spans, sents = segment_by_fixed_size_with_step( test_input, group_size=6, step=3, language='english', keep_tail=False)
                for m, (seg, (a, b)) in enumerate(zip(segments, spans), 1):
                    # Update the input_dict with the current segment
                    input_dict["text"] = seg
                    # Give template, give the dict as input, get the final output dict
                    output_dict = clients_util.generate_with_calling_openai_api(
                        client=openai_client,
                        system_prompt_template_path="prompts/system/system_prompt_translation.jinja",
                        input_prompt_template_path="prompts/extraction/prompt_extraction_with_grammer_text.jinja",
                        input_text_dict=input_dict,
                        model="gpt-5-mini",
                    )
                    # print(output_dict)
                    # print("----------------------------------------------")
                    # pprint(output_dict, indent=2, width=150, sort_dicts=False)
