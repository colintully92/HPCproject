#!/bin/bash -l
#
#SBATCH --nodes=$nnodes
#SBATCH --ntasks-per-node=$nmpiranks

export OMP_NUM_THREADS=24

srun -N 4 -n 4 ./stencil2d-mpiomp.x --nx 128 --ny 128 --nz 64 --num_iter 1024
