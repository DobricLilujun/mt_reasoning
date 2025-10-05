#!/bin/bash
# uv run vllm serve /home/snt/projects_lujun/base_models/gemma-2-2b-it --host 0.0.0.0 --port 1997 --max-model-len 2048 --max-num-seqs 2 --gpu-memory-utilization 0.7

model_vllm=gpt-5
PORT=1997

uv run python scripts/text2grammar_task3.py --model_vllm $model_vllm --port $PORT --sentence_list_size 2 --grammar_size 4 --test_length 1000
uv run python scripts/text2grammar_task3.py --model_vllm $model_vllm --port $PORT --sentence_list_size 2 --grammar_size 6 --test_length 1000
uv run python scripts/text2grammar_task3.py --model_vllm $model_vllm --port $PORT --sentence_list_size 2 --grammar_size 8 --test_length 1000
uv run python scripts/text2grammar_task3.py --model_vllm $model_vllm --port $PORT --sentence_list_size 2 --grammar_size 10 --test_length 1000
uv run python scripts/text2grammar_task3.py --model_vllm $model_vllm --port $PORT --sentence_list_size 2 --grammar_size 12 --test_length 1000
uv run python scripts/text2grammar_task3.py --model_vllm $model_vllm --port $PORT --sentence_list_size 3 --grammar_size 6 --test_length 1000
uv run python scripts/text2grammar_task3.py --model_vllm $model_vllm --port $PORT --sentence_list_size 3 --grammar_size 9 --test_length 1000
uv run python scripts/text2grammar_task3.py --model_vllm $model_vllm --port $PORT --sentence_list_size 3 --grammar_size 12 --test_length 1000
uv run python scripts/text2grammar_task3.py --model_vllm $model_vllm --port $PORT --sentence_list_size 3 --grammar_size 15 --test_length 1000
uv run python scripts/text2grammar_task3.py --model_vllm $model_vllm --port $PORT --sentence_list_size 3 --grammar_size 18 --test_length 1000

uv run python scripts/text2grammar_task4.py --model_vllm $model_vllm --port $PORT
