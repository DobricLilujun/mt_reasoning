import os
import requests

def generate_with_calling_vllm(server_url, model_name, prompt, api_key=None):

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY")

    if "openai" in server_url:
        # server_url = server_url.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            # "temperature": 0.1, # Not Supported
            # "top_p": 0.9,
            "max_completion_tokens": 32768,
            # "frequency_penalty": 0.0,
            # "n": 1,
        }
    else:
        headers = {
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 32768,
            "frequency_penalty": 0.0,
            "n": 1,
        }
    

    resp = requests.post(server_url, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
    data = resp.json()
    if "choices" not in data or not data["choices"]:
        raise RuntimeError(f"Invalid response: {data}")
    
    return data["choices"][0]["message"]["content"], data["choices"][0]["message"]["content"]
