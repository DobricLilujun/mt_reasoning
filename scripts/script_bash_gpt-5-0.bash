#!/bin/bash
# uv run vllm serve /home/snt/projects_lujun/base_models/gemma-2-2b-it --host 0.0.0.0 --port 1997 --max-model-len 2048 --max-num-seqs 2 --gpu-memory-utilization 0.7

model_vllm=gpt-5
PORT=1997

uv run python scripts/text2grammar_task1.py --model_vllm $model_vllm --port $PORT --grammar_list_size 2
uv run python scripts/text2grammar_task1.py --model_vllm $model_vllm --port $PORT --grammar_list_size 3
uv run python scripts/text2grammar_task1.py --model_vllm $model_vllm --port $PORT --grammar_list_size 4
uv run python scripts/text2grammar_task1.py --model_vllm $model_vllm --port $PORT --grammar_list_size 5
uv run python scripts/text2grammar_task1.py --model_vllm $model_vllm --port $PORT --grammar_list_size 6
uv run python scripts/text2grammar_task1.py --model_vllm $model_vllm --port $PORT --grammar_list_size 7
uv run python scripts/text2grammar_task1.py --model_vllm $model_vllm --port $PORT --grammar_list_size 8
uv run python scripts/text2grammar_task1.py --model_vllm $model_vllm --port $PORT --grammar_list_size 9
uv run python scripts/text2grammar_task1.py --model_vllm $model_vllm --port $PORT --grammar_list_size 10

