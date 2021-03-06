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
### READ BINARY DATA HERE - to do ###
#np.fromfile(stuff)

@click.command()
@click.option("--src", type=str, required=True, help="Path to the first field.")
@click.option("--trg", type=str, required=True, help="Path to the second field.")
@click.option(
    "--rtol", type=float, required=False, default=1e-5, help="Relative tolerance."
)
@click.option(
    "--atol", type=float, required=False, default=1e-8, help="Absolute tolerance."
)
def main(src, trg, rtol=1e-5, atol=1e-8): 
    src_f = np.fromfile(src,float) #np.load(src)
    print(np.shape(src_f))
    #src_f = np.array(src_f)
    trg_f = np.fromfile(trg,float) #np.load(trg)
    print(np.shape(trg_f))
    #trg_f = np.array(trg_f)
    truth = np.zeros(len(src_f))
    
    for i in range (0,len(src_f)):
        if np.allclose(src_f[i], trg_f[i], rtol=rtol, atol=atol, equal_nan=True): 
            #src_f[i] == trg_f[i]:
            truth[i] = 1
        else:
            #print('not equal at this point:', i, src_f[i], trg_f[i])
            truth[i] = 0
    
    if np.allclose(src_f, trg_f, rtol=rtol, atol=atol, equal_nan=True):
        print(f"HOORAY! '{src}' and '{trg}' are equal!")
    else:
        print(f"{src} and {trg} are not equal.")
        
    print(truth)
    np.savetxt('equivalence_mpi24.out', truth)


if __name__ == "__main__":
    main()

def read_field_from_file(filename, num_halo=None):
    (rank, nbits, num_halo, nx, ny, nz) = np.fromfile(filename, dtype=np.int32, count=6)
    offset=(3 + rank) * 32 // nbits
    data = np.fromfile(filename, dtype=np.float32 if nbits == 32 else np.float64, \
                       count=nz * ny * nx + offset)
    if rank == 3:
        return np.reshape(data[offset:], (nz, ny, nx))
    else:
        return np.reshape(data[offset:], (ny, nx))

fig, axs = plt.subplots(1, 1, figsize=(12, 4))

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