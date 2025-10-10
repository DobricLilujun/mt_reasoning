#!/bin/bash -l
 
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 7
#SBATCH -G 1
#SBATCH --time=08:00:00
#SBATCH -p gpu
#SBATCH -C volta32
#SBATCH --qos=normal
 
 
# watch nvidia-smi

load_apptainer="module load tools/Apptainer"
singularity_path="/home/users/luli/singularity_images/ollama.sif"
 
pull_ollama="singularity pull ${singularity_path} docker://ollama/ollama:latest"
 
 
apptainer_run_ollama="apptainer run --nv \
    --env OLLAMA_CONTEXT_LENGTH=10000 \
    --env OLLAMA_LOAD_TIMEOUT=120m \
    ${singularity_path}"
 
start_ollama="${load_apptainer}; ${pull_ollama}; ${apptainer_run_ollama};"
 
port_mapping="11434:localhost:11434"
ssh -oStrictHostKeyChecking=no $1 -L $port_mapping $start_ollama
 