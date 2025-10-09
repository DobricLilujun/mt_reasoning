python scripts/eval_performance.py --model_vllm  /home/snt/projects_lujun/base_models/gemma-2-9b-it --port 1997

python scripts/eval_performance.py --model_vllm  gpt-5-mini --port 1997

export TMPDIR=/home/snt/projects_lujun/mt_reasoning/data/tmp
export VLLM_CACHE_ROOT=/home/snt/projects_lujun/mt_reasoning/data/tmp

uv run vllm serve /home/snt/projects_lujun/base_models/gemma-2-9b-it --host 0.0.0.0 --port 1997 --max-model-len 2048 --gpu-memory-utilization 0.8 