#!/usr/bin/env zsh

#SBATCH --job-name=task2        # Name of the job
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=1          # Number of tasks to be launched (usually the number of CPU cores you want)
#SBATCH --gres=gpu:1                 # Number of GPUs
#SBATCH --time=00:01:00              # Wall-clock time limit
#SBATCH --partition=instruction   # Partition or queue. Replace 'YOUR_PARTITION' with the appropriate value.
#SBATCH --output=task2.out    # File to which STDOUT will be written
#SBATCH --error=task2.err     # File to which STDERR will be written

module load nvidia/cuda/11.8
module load gcc/.11.3.0_cuda

# Run the compiled CUDA program
./task2