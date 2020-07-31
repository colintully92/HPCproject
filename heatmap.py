# ******************************************************
#     Program: compare_fields.py
#      Author: Stefano Ubbiali
#       Email: subbiali@phys.ethz.ch
#        Date: 04.06.2020
# Description: Comparing two NumPy arrays
# ******************************************************
import click
import numpy as np
import matplotlib.pyplot as plt

def read_field_from_file(filename, num_halo=None):
    (rank, nbits, num_halo, nx, ny, nz) = np.fromfile(filename, dtype=np.int32, count=6)
    offset=(3 + rank) * 32 // nbits
    data = np.fromfile(filename, dtype=np.float32 if nbits == 32 else np.float64, \
                       count=nz * ny * nx + offset)
    if rank == 3:
        return np.reshape(data[offset:], (nz, ny, nx))
    else:
        return np.reshape(data[offset:], (ny, nx))

fig, axs = plt.subplots(1, 1) #, figsize=(12, 4))

ref_field = read_field_from_file('out_field_mpi.dat')
hybrid_field = read_field_from_file('out_field_mpiomp.dat')
comp_field = hybrid_field - ref_field
    
k_lev = in_field.shape[0] // 2
im1 = axs[0].imshow(comp_field[k_lev, :, :], origin='lower', vmin=-0.1, vmax=1.1);
fig.colorbar(im1, ax=axs[0]);
axs[0].set_title('Comparison field (k = {})'.format(k_lev));

    
    #k_lev = out_field.shape[0] // 2
    #im2 = axs[1].imshow(out_field[k_lev, :, :], origin='lower', vmin=-0.1, vmax=1.1);
    #fig.colorbar(im2, ax=axs[1]);
    #axs[1].set_title('Final result (k = {})'.format(k_lev));
    
plt.savefig('test.png')