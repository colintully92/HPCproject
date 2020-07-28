#!/bin/bash

#This script launches all the jobs for our comparison: the mpi base version, mpiomp version, and compare script
#See notes below for running mpiomp versions with single or multiple nodes

#General commands to run on daint
module unload PrgEnv-gnu
module load PrgEnv-cray
make VERSION=mpi
make VERSION=mpiomp

declare -x nnodes=3 #set number of nodes (srun -N #)
declare -x nmpiranks=24 # set number of mpi ranks (srun -n #), ranks spread over all nodes
### For running mpiomp version
declare -x nthreads=3 # Set the number of openMP threads 

export OMP_NUM_THREADS=$nthreads
    declare -x ncores=$nthreads
  #sets upper bound on number of threads
  #cannot use more threads then there are cores available (???)
    if [ $nthreads -gt 24 ] ; then
        ncores=24
    fi

if [ $nnodes -eq 1 ]; then 
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
else
    echo "running on multiple nodes"
    #for running on multiple nodes
    #requires slurm run jobscript
    #srun -N $nnodes -n $nmpiranks -c $ncores ./stencil2d-mpiomp.x --nx 128 --ny 128 --nz 64 --num_iter 1024
    sbatch -C gpu -N $nnodes -n $nmpmpiranks ./stencil2d-mpi.x --nx 128 --ny 128 --nz 64 --num_iter 1024
    sbatch -C gpu -N $nnodes -n $nmpiranks -c $ncores ./stencil2d-mpiomp.x --nx 128 --ny 128 --nz 64 --num_iter 1024
    #./run_job.sh
    #echo "running stencil2d-mpi.F90 ..."
    #srun -N $nnodes -n $nmpiranks ./stencil2d-mpi.x --nx 128 --ny 128 --nz 64 --num_iter 1024
    #echo "running stencil2d-mpiomp.F90 ..."
    #srun -N $nnodes -n $nmpiranks -c $ncores ./stencil2d-mpiomp.x --nx 128 --ny 128 --nz 64 --num_iter 1024
fi

# compare output against control data
echo "running compare_fields.py ..."
python compare_fields.py --src="out_field_mpi.dat" --trg="out_field_mpiomp.dat"
