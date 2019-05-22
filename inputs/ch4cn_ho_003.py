import pickle
import numpy as np
import sklearn.pipeline
from qml import qmlearn
import h5py
from random import shuffle
from sklearn.model_selection import train_test_split


# Create data
data = h5py.File("../datasets/dftb_vr_cc.hdf5")

xyz = np.array(data.get("xyz"))
ene = np.array(data.get("ene"))
zs = np.array(data.get("zs"), dtype=np.int32)

# Corresponds approximately to the energy of the dissociated products
ref_energy = -133.2 * 2625.5

# Load the indices of the test trajectory
traj_data = np.load("../outputs/find_traj_idx.npz")
idx_traj = traj_data["arr_0"]

# Finding train indices
idx_full = list(range(ene.shape[0]))

# Removing the indices corresponding to the trajectory
idx_no_traj = []
for item in idx_full:
    if not item in idx_traj:
        idx_no_traj.append(item)

idx_no_traj = np.asarray(idx_no_traj)

# Leaving 10% of the data for testing
idx_train, idx_test = train_test_split(idx_no_traj, test_size=0.1, shuffle=True)

print("%i samples for training, %i samples for testing plus a trajectory of %i samples"
        %(len(idx_train),len(idx_test),len(idx_traj)))

# Saving the indices of train/test
np.savez("../outputs/ch4cn_ho_003.npz", idx_train, idx_test)

# Making the data object
data = qmlearn.Data()
data.coordinates = xyz[idx_train]
data.nuclear_charges = zs[idx_train]
data._set_ncompounds(len(data.nuclear_charges))
data.natoms = np.asarray([len(data.nuclear_charges[0])]*len(data.nuclear_charges))
ene = ene - ref_energy
data.set_energies(ene[idx_train])

# Create model
estimator = sklearn.pipeline.make_pipeline(
                qmlearn.representations.GlobalSLATM(data),
                qmlearn.models.NeuralNetwork(hl3=0)
                )

# Saving the model and training on all the samples in the model
pickle.dump(estimator, open('../outputs/ch4cn_ho_003.pickle', 'wb'))
indices = np.arange(len(idx_train))
with open('../outputs/ch4cn_ho_003.csv', 'w') as f:
    for i in indices:
        f.write('%s\n' % i)
