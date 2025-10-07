import os
import requests
import time
import json
from typing import Dict, Any
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from torch import chunk



def render_prompt(template_path: str, **vars):
    p = Path(template_path)
    env = Environment(
        loader=FileSystemLoader(str(p.parent)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tmpl = env.get_template(p.name)
    return tmpl.render(**vars)

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
            "max_completion_tokens": 25000,
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
            "max_tokens": 25000,
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


def generate_with_calling_openai_api(
                        client,
                        system_prompt_template_path: str,         
                        input_prompt_template_path: str,
                        input_text_dict: Dict[str, str] = None,
                        model: str = "gpt-5-mini",
                        temperature: float = 0.1,
                        max_retries: int = 5,
                        initial_backoff: float = 1.0) -> Dict[str, Any]:
    
    if not input_text_dict or not isinstance(input_text_dict, dict):
        raise ValueError("input_text_dict must be a non-empty dict with at least one text-like field.")
    
    ctx = dict(input_text_dict)

    system_prompt = render_prompt(system_prompt_template_path)
    input_prompt = render_prompt(input_prompt_template_path, **ctx)

    for attempt in range(max_retries):
        try:
            messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_prompt},
            ]
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages,
                response_format={"type": "json_object"}, 
            )
            content = resp.choices[0].message.content
            return json.loads(content), messages
        except Exception as e:
            sleep_s = initial_backoff * (2 ** attempt) + 0.1 * (attempt)
            time.sleep(sleep_s)
            if attempt == max_retries - 1:
                raise


## Calling API with extra body params for vLLM server
def generate_with_calling_api(
        client,
        system_prompt_template_path: str,
        input_prompt_template_path: str,
        input_text_dict: Dict[str, str] = None,
        model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        temperature: float = 1.0,
        max_retries: int = 5,
        initial_backoff: float = 1.0,
        vllm_extra: Dict[str, Any] = None 
    ) -> Dict[str, Any]:

    if not input_text_dict or not isinstance(input_text_dict, dict):
        raise ValueError("input_text_dict must be a non-empty dict with at least one text-like field.")

    ctx = dict(input_text_dict)

    system_prompt = render_prompt(system_prompt_template_path)
    input_prompt = render_prompt(input_prompt_template_path, **ctx)

    extra_body = vllm_extra or {}

    for attempt in range(max_retries):
        try:
            if "gemma" in model.lower():
                messages = [
                    {"role": "user", "content": input_prompt},
                ]
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_prompt},
                ]
            if "gpt" in model.lower():
                # No extra body for gpt models
                resp = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=messages,
                    response_format={"type": "json_object"},
                    logprobs=extra_body.get("logprobs", None),
                )
            else:
                # No extra body for gpt models
                resp = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=messages,
                    response_format={"type": "json_object"},
                    logprobs=extra_body.get("logprobs", None),
                    top_logprobs=extra_body.get("top_logprobs", None),

                )
            content = resp.choices[0].message.content
            log_prob = resp.choices[0].logprobs
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError()
            
            return data, messages, log_prob
        except (json.JSONDecodeError, Exception) as e:
            if isinstance(e, json.JSONDecodeError):
                print(f"Json Decode Error: {e}")
            sleep_s = initial_backoff * (2 ** attempt) + 0.1 * attempt
            time.sleep(sleep_s)
            if attempt == max_retries - 1:
                raise