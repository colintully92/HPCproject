#!/bin/bash -l
#
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2

echo "Number of nodes: $SLURM_JOB_NUM_NODES"

# nodes: $SLURM_JOB_NUM_NODES
# tasks: $SLURM_NTASKS 
# tasks-per-node: $SLURM_NTASKS_PER_NODE 
# cpus-per-task: $SLURM_CPUS_PER_TASK

srun ./stencil2d-mpi.x --nx 128 --ny 128 --nz 64 --num_iter 1024

srun ./stencil2d-mpiomp.x --nx 128 --ny 128 --nz 64 --num_iter 1024

