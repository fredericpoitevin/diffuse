#!/bin/bash
#
#SBATCH --job-name=maskbragg
#SBATCH --output=maskbraggXXXibatchXXX.out
#SBATCH --error=maskbraggXXXibatchXXX.err
#
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres gpu:4
#
srun ./run_apply_bragg.sh 
