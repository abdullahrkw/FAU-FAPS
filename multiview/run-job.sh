#!/bin/bash -l
# start 2 MPI processes
#SBATCH --ntasks-per-node=2
# allocate nodes for hh:mm:ss
#SBATCH --time=24:00:00
# allocated one GPU (type not specified)
#SBATCH --gres=gpu:1
# job name
#SBATCH --job-name=FAU-FAPS-multiview
# do not export environment variables
#SBATCH --export=NONE
# do not export environment variables
unset SLURM_EXPORT_ENV
source /home/hpc/iwfa/iwfa018h/.bashrc
conda activate FAPS
python trainer.py
