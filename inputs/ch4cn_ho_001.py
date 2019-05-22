import pickle
import numpy as np
import sklearn.pipeline
from qml import qmlearn
import h5py
from random import shuffle

# Create data
dataset = h5py.File("../datasets/dftb_vr_cc.hdf5")

# Corresponds approximately to the energy of the dissociated products
ref_energy = -133.2 * 2625.5


