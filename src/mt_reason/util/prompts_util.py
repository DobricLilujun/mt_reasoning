import pandas as pd
from importlib.resources import files, as_file
import requests
import os
import yaml


# Load ISO 639 DataFrame
def load_iso639_df() -> pd.DataFrame:
    resource = files("mt_reason.source") / "iso-639-3-basic.csv"
    resource_manifest = files("mt_reason") / "manifest.yaml"
    local_path = str(resource)
    
    # Check if the file exists locally
    if not os.path.isfile(local_path):
        # Read manifest.yaml to get download URL
        with open(resource_manifest, "r", encoding="utf-8") as f:
            manifest = yaml.safe_load(f)
        url = None
        for file_info in manifest.get("files", []):
            if file_info.get("name") == "iso_639_3_basic_csv":
                url = file_info.get("url")
                break
        if url is None:
            raise RuntimeError("Download URL for iso_639_3_basic_csv not found in manifest.yaml")
        print(f"File not found locally, downloading from {url}...")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        download_file(url, local_path)

    with as_file(resource) as p:
        df = pd.read_csv(p, sep="|", header=None, dtype=str, encoding="utf-8")
    df.columns = ["iso_639_2B", "iso_639_2T", "iso_639_1", "name_en", "name_fr"]
    return df

# Translate prompt Definition
def translate_prompt(src_text: str, src_lang_code: str, tgt_lang_code: str) -> str:
    df_iso639 = load_iso639_df()
    src_lang = df_iso639[df_iso639["iso_639_2B"] == src_lang_code]["name_en"].values[0]
    tgt_lang = df_iso639[df_iso639["iso_639_2B"] == tgt_lang_code]["name_en"].values[0]
    prompt = f'Translate this sentence from {src_lang} language to {tgt_lang} language: "{src_text}"'
    return prompt



def download_file(url: str, local_path: str, retries=3, timeout=30):
    for _ in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(response.content)
            return
        except Exception as e:
            print(f"Download failed, retrying... Error: {e}")
    raise RuntimeError(f"Failed to download the file from {url}")
