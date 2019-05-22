import glob
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style("white")
sns.set_context("poster")

if os.path.exists("../datasets/dftb_vr.hdf5"):
    data = h5py.File("../datasets/dftb_vr.hdf5", "r")
    xyz = np.array(data.get("xyz"))
    ene = np.array(data.get("ene"))
    zs = np.array(data.get("zs"), dtype=np.int32)

else:
    files = glob.glob("/Volumes/Transcend/data_sets/dftb/*.xyz")

    xyz = []
    zs = []
    ene = []

    elems = {'H':1, 'C':6, 'N':7}

    for f in files:
        f_in = open(f, "r")

        zs_mol = []
        xyz_mol = []

        for line in f_in:

            if 'Energy' in line:
                idx_start = line.find("Energy: ") + len("Energy: ")
                idx_end = line.find("(hartree)") - 1
                ene_ha = float(line[idx_start:idx_end])
                ene.append(ene_ha * 2625.5)
            elif 'C' in line or 'H' in line or 'N' in line:
                linesplit = line.split("\t")
                zs_mol.append(elems[linesplit[0]])
                xyz_atom = []
                for i in range(3):
                    xyz_atom.append(float(linesplit[i+1]))
                xyz_mol.append(xyz_atom)

        xyz.append(xyz_mol)
        zs.append(zs_mol)

    xyz = np.asarray(xyz)
    zs = np.asarray(zs)
    ene = np.asarray(ene)

    hdf5_file = h5py.File("../datasets/dftb_vr.hdf5", mode='w')

    hdf5_file.create_dataset("xyz", xyz.shape, np.float32, data=xyz)
    hdf5_file.create_dataset("ene", ene.shape, np.float32, data=ene)
    hdf5_file.create_dataset("zs", zs.shape, np.int32, data=zs)

    hdf5_file.close()

print(xyz.shape, zs.shape, ene.shape)
nuc_char = {1:'H', 6:'C', 7:'N'}

idx = []
for samp in range(ene.shape[0]):
    idx_c = np.where(zs[samp] == 6.0)[0]
    assert len(idx_c) == 2
    cc_dist = np.linalg.norm(xyz[samp, idx_c[0]]-xyz[samp, idx_c[1]])

    if cc_dist <= 2 and  ene[samp] <= -18900:
        idx.append(samp)

print(len(idx))

traj_file = open("../outputs/analyse_dftb_2.xyz", "w")
for i in idx:
    traj_file.write(str(len(xyz[i])))
    traj_file.write("\n\n")

    for atm in range(len(xyz[i])):
        traj_file.write(nuc_char[zs[i, atm]] + "\t")
        for coord in range(3):
            traj_file.write(str(xyz[i][atm][coord])+"\t")
        traj_file.write("\n")


traj_file.close()