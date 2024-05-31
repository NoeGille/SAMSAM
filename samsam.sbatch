#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=samsam            # Name of the job 
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=samsam-output.log   # Log-file (important!)
#SBATCH --cpus-per-task=2           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=20G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:2                # Number of GPU's
#SBATCH --time=48:00:00              # Max execution time

# Activate your Anaconda environment
conda activate samsam

# Run your Python script
python train.py
