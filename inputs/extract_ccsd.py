# Making the CCSD data into a hdf5 file
import sys
sys.path.insert(0,'/Volumes/Transcend/repositories/YAMLP/SciFlow')
import ImportData as imp
import os
import h5py
import numpy as np

if os.path.exists("../datasets/dftb_vr_cc.hdf5"):
    data = h5py.File("../datasets/dftb_vr_cc.hdf5", "r")
    xyz = np.array(data.get("xyz"))
    ene = np.array(data.get("ene"))
    zs = np.array(data.get("zs"), dtype=np.int32)
else:

    datasets_X = "/Volumes/Transcend/repositories/trainingNN/dataSets/XYQ/X_b3lyp.csv"
    datasets_y = "/Volumes/Transcend/repositories/trainingNN/dataSets/XYQ/Y_b3lyp.csv"

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

    hdf5_file = h5py.File("../datasets/dftb_vr_cc.hdf5", mode='w')

    hdf5_file.create_dataset("xyz", xyz.shape, np.float32, data=xyz)
    hdf5_file.create_dataset("ene", ene.shape, np.float32, data=ene)
    hdf5_file.create_dataset("zs", zs.shape, np.int32, data=zs)

    hdf5_file.close()
