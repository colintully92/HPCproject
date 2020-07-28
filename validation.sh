#!/bin/bash

#This script launches all the jobs for our comparison: the mpi base version, mpiomp version, and compare script
#See notes below for running mpiomp versions with single or multiple nodes

#OPTIONAL clean before starting 
make clean

#General commands to run on daint
module unload PrgEnv-gnu
module load PrgEnv-cray
make VERSION=mpi
make VERSION=mpiomp

declare -x nnodes=4 #set number of nodes (srun -N #)
declare -x nmpiranks=4 # set number of mpi ranks (srun -n #), ranks spread over all nodes
### For running mpiomp version
declare -x nthreads=24 # Set the number of openMP threads 

export OMP_NUM_THREADS=$nthreads
    declare -x ncores=$nthreads
  #sets upper bound on number of threads
  #cannot use more threads then there are cores available (???)
    if [ $nthreads -gt 24 ] ; then
        ncores=24
    fi

if [ $nnodes -eq 0 ]; then 
    # generate reference data
    echo "running stencil2d-mpi.F90 ..."
    #cd ../HPC4WC/day3 && \
    #srun stencil2d.py --nx=128 --ny=128 --nz=64 --num_iter=${num_iter} && \
    srun -N $nnodes -n $nmpiranks ./stencil2d-mpi.x --nx 128 --ny 128 --nz 64 --num_iter 1024
  #cd ../../Project || exit
    # run the programm to validate
    echo "running stencil2d-mpiomp.F90 ..."
    #for running on single node
    srun -N $nnodes -n $nmpiranks -c $ncores ./stencil2d-mpiomp.x --nx 128 --ny 128 --nz 64 --num_iter 1024
    echo "running compare_fields.py ..."
    python compare_fields.py --src="out_field_mpi.dat" --trg="out_field_mpiomp.dat"
else
    echo "running on multiple nodes"
    #for running on multiple nodes
    #requires slurm run jobscript
   
    #Run mpi only version
    sbatch -C gpu run_mpi.sh
    #Run hybrid version
    sbatch -C gpu run_hybrid.sh
    squeue -u course40
fi

# compare output against control data

#### NOTES for Slurm ####
# nodes = --nodes 
# mpi ranks = --nodes * --ntasks-per-node
# openMP threads = export OMP_NUM_THREADS