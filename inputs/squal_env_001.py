from qml.representations import generate_acsf
import numpy as np
import h5py
import os

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set_style("white")
sns.set_context("poster")

def acsfy(xyz, classes, acsf_params):
    """
    Generates the ACSF from the coordinates, the atom type and the hyper-parameters given.
    """

    elements = np.unique(classes)
    elements_no_zero = np.ma.masked_equal(elements, 0).compressed()

    representation = []

    for i in range(xyz.shape[0]):

        g = generate_acsf(coordinates=xyz[i], elements=elements_no_zero, gradients=False, nuclear_charges=classes[i],
                          rcut=acsf_params['rcut'],
                          acut=acsf_params['acut'],
                          nRs2=acsf_params['nRs2'],
                          nRs3=acsf_params['nRs3'],
                          nTs=acsf_params['nTs'],
                          eta2=acsf_params['eta'],
                          eta3=acsf_params['eta'],
                          zeta=acsf_params['zeta'],
                          bin_min=0.8)

        # Hotfix t make sure the representation is single precision
        single_precision_g = g.astype(dtype=np.float32)
        del g

        representation.append(single_precision_g)

    return np.asarray(representation)

def reshape_trim(acsf, classes):
    # Indices of carbon atoms in the first sample
    c_idx = np.where(classes[0] == 6)[0]
    # Number of carbon atoms
    n_c = len(c_idx)
    # Writing the indices so one can figure out which atom in the molecules the carbon was
    atom_idx = np.tile(c_idx, acsf.shape[0])
    # Writing the indices so one can figure out from which molecule a carbon came from
    mol_idx = np.repeat(np.asarray(range(acsf.shape[0])), n_c)
    # Concatenating the acsf and the classes for all atoms in a molecule
    acsf = np.reshape(acsf, (acsf.shape[0]*acsf.shape[1], acsf.shape[-1]))
    classes = np.reshape(classes, (classes.shape[0] * classes.shape[1], ))

    flat_c_idx= np.where(classes==6)[0]

    return acsf[flat_c_idx], mol_idx, atom_idx

def write_vmd(xyz, zs, bad_c_idx):

    dict = {1:"H", 6:"C", 7:"N", 2:"P"}

    f = open("../outputs/squal_env_001/squal_analysis.xyz", "w")
    f.write(str(len(zs)))
    f.write("\n\n")

    for i in range(xyz.shape[0]):
        if i in bad_c_idx:
            zs[i] = 2
        f.write(dict[zs[i]])
        f.write("\t")

        for k in range(3):
            f.write(str(xyz[i][k]))
            f.write("\t")
        f.write("\n")
    f.close()

# Creating output dir
if not os.path.exists("../outputs/squal_env_001/"):
    os.makedirs("../outputs/squal_env_001/")

# Getting the dataset
data_isopentane = h5py.File("../datasets/NN-sq/isopentane_cn_dft.hdf5", "r")
data_squal = h5py.File("../datasets/NN-sq/squalane_cn_dft.hdf5", "r")

# Squalane data
xyz_squal = np.array(data_squal.get("xyz"))
zs_squal = np.array(data_squal.get("zs"), dtype=np.int32)

# Isopentane (secondary abstraction)
idx_isopentane = np.asarray(data_isopentane.get('traj_idx'), dtype=int)
idx_isopentane_traj = np.where(idx_isopentane == 1)[0]

xyz_isopentane = np.array(data_isopentane.get("xyz"))[idx_isopentane_traj]
zs_isopentane = np.array(data_isopentane.get("zs"), dtype=np.int32)[idx_isopentane_traj]

# Generating all the representations
n_basis = 14
r_min = 0.8
r_cut = 3.248470148281216
tau = 1.6110162523935854
eta = 4 * np.log(tau) * ((n_basis-1)/(r_cut - r_min))**2
zeta = - np.log(tau) / (2*np.log(np.cos(np.pi/(4*n_basis-4))))

acsf_params={"nRs2":n_basis, "nRs3":n_basis, "nTs":n_basis, "rcut":r_cut, "acut":r_cut, "zeta":zeta, "eta":eta}

n_samples_squal = 10
n_samples_isopentane = -1
acsf_isopentane = acsfy(xyz_isopentane[:n_samples_isopentane], zs_isopentane[:n_samples_isopentane], acsf_params)
acsf_squal = acsfy(xyz_squal[:n_samples_squal], zs_squal[:n_samples_squal], acsf_params)
print("The acsf for isopentane and squalane have shape %s and %s respectively." % (str(acsf_isopentane.shape), str(acsf_squal.shape)))

# Removing all the non-carbon atoms
acsf_isopentane_c, _, _ = reshape_trim(acsf_isopentane, zs_isopentane[:n_samples_isopentane])
acsf_squal_c, mol_idx, atom_idx = reshape_trim(acsf_squal, zs_squal[:n_samples_squal])
print("The shape of the trimmed acsf are %s and %s for isopentane and squalane respectively" % (str(acsf_isopentane_c.shape), str(acsf_squal_c.shape)))
print(mol_idx.shape, atom_idx.shape)

# Comparing the carbon atoms from squalane to isopentane
bad_represented_c = []
diff_for_hist = []
for j in range(acsf_squal_c.shape[0]): # Looping over all squalane carbons
    diff_man=[]

    for i in range(acsf_isopentane_c.shape[0]): # Looping over all isopentane carbons
        diff_man.append(np.sum(np.abs(acsf_isopentane_c[i] - acsf_squal_c[j])))
    min_d = min(diff_man)
    diff_for_hist.append(min_d)

    if min_d  >= 6:
        bad_represented_c.append(atom_idx[j])

write_vmd(xyz_squal[0], zs_squal[0], list(set(bad_represented_c)))

# Getting the histograms for debugging purposes
bins_man = np.arange(0, max(diff_for_hist), 0.5)
hist_man, bin_man = np.histogram(diff_for_hist, bins=bins_man, density=False)

part_fig_2, part_ax_2 = plt.subplots(figsize=(11, 10))
part_ax_2.plot(bins_man[1:], hist_man, label="Isopentane")
part_ax_2.set(xlabel="Manhattan distance", ylabel="Occurrences")
part_ax_2.legend()
part_fig_2.savefig("../outputs/squal_env_001/acsf_man.png", dpi=200)
