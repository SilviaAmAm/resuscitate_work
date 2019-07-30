from qml.aglaia.aglaia import MRMP
import numpy as np
import h5py
import time
import os
import tensorflow as tf

if not os.path.exists("../outputs/ch4cn_batchsize_001/"):
    os.makedirs("../outputs/ch4cn_batchsize_001/")

# Create data
with h5py.File("../datasets/dftb_vr_cc.hdf5") as data:
    xyz = np.array(data.get("xyz"))
    ene = np.array(data.get("ene"))
    zs = np.array(data.get("zs"), dtype=np.int32)

# Corresponds approximately to the energy of the dissociated products
ref_energy = -133.2 * 2625.5
ene = ene-ref_energy

# Load the indices of the training/test set and the test trajectory
traj_data = np.load("../outputs/find_traj_idx.npz")
idx_traj = traj_data["arr_0"]
train_data = np.load("../outputs/ch4cn_ho_003.npz")
idx_train, idx_test = train_data["arr_0"], train_data["arr_1"]

print(len(idx_train))

batch_sizes = [5, 50, 100, 500, 1000, 3000]
scores = []
wall_times = []
grad_updates = []

for bs in batch_sizes:
    # Creating the model
    tb_subdir = "../outputs/ch4cn_batchsize_001/tb_" + str(bs)
    estimator = MRMP(iterations=2401, l1_reg=3.7260078742338124e-06, l2_reg=2.1321843097427243e-07, learning_rate=0.0004,
                     representation_name='unsorted_coulomb_matrix', store_frequency=25, tensorboard=True, tensorboard_subdir=tb_subdir, hidden_layer_sizes=(32,298,), batch_size=bs)
    
    estimator.set_properties(ene)
    estimator.generate_representation(xyz, zs, method='fortran')
    start = time.time()    
    estimator.fit(idx_train)
    end = time.time()

    wall_times.append(end-start)
    score = estimator.score(idx_test)
    scores.append(score)
    grad_updates.append(int(len(idx_train)/bs))

    tf.reset_default_graph()
    del estimator

    print("Finished with batch size %i, score: %s, time: %s" % (bs, str(score), str(end-start)))

np.savez("../outputs/ch4cn_batchsize_001.npz", batch_sizes, scores, wall_times, grad_updates)
