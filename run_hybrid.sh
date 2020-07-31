#!/bin/bash -l
#
#SBATCH --time=00:30:00
#SBATCH --ntasks=20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --ntasks-per-core=2

export OMP_NUM_THREADS=1
#Add opneMP tuning here
#export OMP_SCHEDULE="DYNAMIC"
#echo "Loop scheduling: $OMP_SCHEDULE" 

srun ./stencil2d-mpiomp.x --nx 128 --ny 128 --nz 64 --num_iter 1024

#NOTES
#MPI ranks = ntasks
#divide mpi ranks over nodes = ntasks-per-node