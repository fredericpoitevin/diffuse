#!/bin/bash
#
#SBATCH --job-name=gdup
#SBATCH --output=gdup%j.out
#SBATCH --error=gdup.%j.err
#
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres gpu:1

srun ./run.sh
