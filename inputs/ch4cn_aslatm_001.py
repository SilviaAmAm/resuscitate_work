from qml.aglaia.aglaia import ARMP
import numpy as np
import h5py

# Create data
data = h5py.File("../datasets/dftb_vr_cc.hdf5")

xyz = np.array(data.get("xyz"))
ene = np.array(data.get("ene"))
zs = np.array(data.get("zs"), dtype=np.int32)

# Corresponds approximately to the energy of the dissociated products
ref_energy = -133.2 * 2625.5
ene = ene-ref_energy

# Load the indices of the training/test set and the test trajectory
traj_data = np.load("../outputs/find_traj_idx.npz")
idx_traj = traj_data["arr_0"]
train_data = np.load("../outputs/ch4cn_ho_005.npz")
idx_train, idx_test = train_data["arr_0"], train_data["arr_1"]

# Creating the model
estimator = ARMP(iterations=2493, l1_reg=3.130673727016551e-05, l2_reg=8.559023858043487e-07, learning_rate=0.003258630161819941, representation_name='slatm', tensorboard=False, hidden_layer_sizes=(91,36,), batch_size=99)

estimator.set_properties(ene)
estimator.generate_representation(xyz, zs, method='fortran')

estimator.fit(idx_train)
estimator.save_nn("../outputs/ch4cn_aslatm_001/")

ene_pred = estimator.predict(idx_test)
ene_pred_traj = estimator.predict(idx_traj)

np.savez("../outputs/ch4cn_aslatm_001.npz", ene_pred, ene[idx_test], ene_pred_traj, ene[idx_traj])

print(estimator.score(idx_test), np.std(ene_pred-ene[idx_test]))
print(estimator.score(idx_traj), np.std(ene_pred_traj-ene[idx_traj]))
