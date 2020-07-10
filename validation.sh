#!/bin/bash

#num_iter=1024

# generate reference data
echo "running stencil2d-mpi.F90 ..."
#cd ../HPC4WC/day3 && \
  #srun stencil2d.py --nx=128 --ny=128 --nz=64 --num_iter=${num_iter} && \
srun -n 12 ./stencil2d-mpi.x+orig --nx 128 --ny 128 --nz 64 --num_iter 1024
  #cd ../../Project || exit

# run the programm to validate
echo "running stencil2d-mpiomp.F90 ..."
#srun stencil2d-cupy.py --nx=128 --ny=128 --nz=64 --num_iter=${num_iter}
srun -n 12 ./stencil2d-mpiomp.x+orig --nx 128 --ny 128 --nz 64 --num_iter 1024

# compare output againts control data
#echo "running compare_fields.py ..."
#python compare_fields.py --src="out_field.dat" --trg="out_field.dat"
