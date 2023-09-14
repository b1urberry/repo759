#!/usr/bin/env zsh

#SBATCH --partition=instruction
#SBATCH --time=00-00:01:00
#SBATCH --ntasks=1

#SBATCH --job-name=FirstSlurm
#SBATCH --cpus-per-task=2
#SBATCH --output=FirstSlurm.out
#SBATCH --error=FirstSlurm.err

cd $SLURM_SUBMIT_DIR

hostname