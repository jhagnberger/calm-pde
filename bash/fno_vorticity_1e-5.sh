#!/bin/bash
#SBATCH --partition=gpu,dgx
#SBATCH --time=8:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

source ~/miniforge3/etc/profile.d/conda.sh
conda activate calm_pde
cd ../calm_pde

python3 train.py --config-name=2d_fno_1e-5 ++experiment.random_seed=3407