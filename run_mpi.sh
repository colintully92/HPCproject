#!/bin/bash -l
#
#SBATCH --time=00:30:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=24

echo "Number of nodes: $SLURM_JOB_NUM_NODES"

srun ./stencil2d-mpi.x --nx 128 --ny 128 --nz 64 --num_iter 1024

#srun ./stencil2d-mpiomp.x --nx 128 --ny 128 --nz 64 --num_iter 1024

