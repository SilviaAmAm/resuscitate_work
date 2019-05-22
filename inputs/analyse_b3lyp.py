import sys
sys.path.insert(0,'/Volumes/Transcend/repositories/YAMLP/SciFlow')
import ImportData as imp
import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style("white")
sns.set_context("poster")

if os.path.exists("../datasets/dftb_vr_b3lyp.hdf5"):
    data = h5py.File("../datasets/dftb_vr_b3lyp.hdf5", "r")
    xyz = np.array(data.get("xyz"))
    ene = np.array(data.get("ene"))
    zs = np.array(data.get("zs"), dtype=np.int32)
else:

    datasets_X = "/Volumes/Transcend/repositories/trainingNN/dataSets/XYQ/X_cc.csv"
    datasets_y = "/Volumes/Transcend/repositories/trainingNN/dataSets/XYQ/Y_cc.csv"

    X = imp.loadX(datasets_X)
    y = imp.loadY(datasets_y)

    zs = np.zeros((len(X), int(len(X[0])/4)))
    xyz = np.zeros((len(X), int(len(X[0])/4), 3))
    ene = np.zeros((len(X),))

    elems = {'H':1, 'C':6, 'N':7}

    for samp in range(len(X)):
        for ii in range(len(X[samp])):
            if ii%4 == 0:
                zs[samp, int(ii/4)] = int(elems[X[samp][ii]])
            else:
                xyz[samp, int(ii/4), ii%4-1] = X[samp][ii]

            ene[samp] = y[samp]*2625.5

    hdf5_file = h5py.File("../datasets/dftb_vr_b3lyp.hdf5", mode='w')

    hdf5_file.create_dataset("xyz", xyz.shape, np.float32, data=xyz)
    hdf5_file.create_dataset("ene", ene.shape, np.float32, data=ene)
    hdf5_file.create_dataset("zs", zs.shape, np.int32, data=zs)

    hdf5_file.close()

elems = {'H':1, 'C':6, 'N':7}
nuc_char = {1:'H', 6:'C', 7:'N'}

# Calculating the CC distance and checking if the energy is higher than -347700 kJ/mol
idx = []
for samp in range(ene.shape[0]):
    idx_c = np.where(zs[samp] == 6.0)[0]
    assert len(idx_c) == 2
    cc_dist = np.linalg.norm(xyz[samp, idx_c[0]]-xyz[samp, idx_c[1]])

    if cc_dist <= 2 and  ene[samp] <= -347700:
        idx.append(samp)

traj_file = open("../outputs/analyse_b3lyp_3.xyz", "w")
for i in idx:
    traj_file.write(str(len(xyz[i])))
    traj_file.write("\n\n")

    for atm in range(len(xyz[i])):
        traj_file.write(nuc_char[zs[i, atm]] + "\t")
        for coord in range(3):
            traj_file.write(str(xyz[i][atm][coord])+"\t")
        traj_file.write("\n")


traj_file.close()
