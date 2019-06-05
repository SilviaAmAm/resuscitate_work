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
train_data = np.load("../outputs/ch4cn_ho_007.npz")
idx_train, idx_test = train_data["arr_0"], train_data["arr_1"]

# Creating the model
# ACSF parameters
n_basis = 15
r_min = 0.8
r_cut = 4.1
tau = 2
eta = 4 * np.log(tau) * ((n_basis-1)/(r_cut - r_min))**2
zeta = - np.log(tau) / (2*np.log(np.cos(np.pi/(4*n_basis-4))))

acsf_params={"nRs2":n_basis, "nRs3":n_basis, "nTs":n_basis, "rcut":r_cut, "acut":r_cut, "zeta":zeta, "eta":eta}

estimator = ARMP(iterations=349, l1_reg=9.56492700994204e-07, l2_reg=5.561300410806235e-07, learning_rate=7.741795296661123e-05, representation_name='acsf', tensorboard=False, hidden_layer_sizes=(70,104,), batch_size=124, representation_params=acsf_params)

estimator.set_properties(ene)
estimator.generate_representation(xyz, zs, method='fortran')

estimator.fit(idx_train)
estimator.save_nn("../outputs/ch4cn_acsf_001/")

ene_pred = estimator.predict(idx_test)
ene_pred_traj = estimator.predict(idx_traj)

np.savez("../outputs/ch4cn_acsf_001.npz", ene_pred, ene[idx_test], ene_pred_traj, ene[idx_traj])

print(estimator.score(idx_test), np.std(ene_pred-ene[idx_test]))
print(estimator.score(idx_traj), np.std(ene_pred_traj-ene[idx_traj]))
