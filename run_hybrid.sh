#!/bin/bash -l
#
#SBATCH --time=00:30:00
#SBATCH --ntasks=80
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=8

export OMP_NUM_THREADS=3
#INSERT OpenMP tuning here

echo "Number of nodes: $SLURM_JOB_NUM_NODES"

srun ./stencil2d-mpiomp.x --nx 128 --ny 128 --nz 64 --num_iter 1024

#NOTES
#MPI ranks = ntasks
#divide mpi ranks over nodes = ntasks-per-node