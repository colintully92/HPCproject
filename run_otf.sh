#!/bin/bash -l
#
#SBATCH --time=00:30:00
#SBATCH --ntasks=96
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=8
#SBATCH --ntasks-per-core=2

export OMP_NUM_THREADS=3
#Add opneMP tuning here
#export OMP_SCHEDULE="DYNAMIC"
echo "On-the-fly" 

#srun ./stencil2d-mpiomp.x --nx 64 --ny 64 --nz 32 --num_iter 128
srun ./stencil2d-mpiomp-otf.x --nx 128 --ny 128 --nz 64 --num_iter 1024

#NOTES
#MPI ranks = ntasks
#divide mpi ranks over nodes = ntasks-per-node