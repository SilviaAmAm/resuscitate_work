import h5py
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style("white")
sns.set_context("poster")

data = h5py.File("../datasets/dftb_vr_cc.hdf5", "r")
xyz = np.array(data.get("xyz"))
ene = np.array(data.get("ene"))
zs = np.array(data.get("zs"), dtype=np.int32)

elems = {'H':1, 'C':6, 'N':7}
nuc_char = {1:'H', 6:'C', 7:'N'}

# Calculating the CC distance and checking if the energy is higher than -347700 kJ/mol
idx = []
for samp in range(ene.shape[0]):
    idx_c = np.where(zs[samp] == 6.0)[0]
    assert len(idx_c) == 2
    cc_dist = np.linalg.norm(xyz[samp, idx_c[0]] - xyz[samp, idx_c[1]])

    if samp >= 12887 and samp <= 13042:
            idx.append(samp)

refined_idx_1 = []
to_exclude = [0,7,18,29,38,41,49,60,63,64,65,69,73,82,92,93,104,114,123,134,140,141,149]

for ii,i in enumerate(idx):
    if not ii in to_exclude:
        refined_idx_1.append(i)

to_exclude = [46,53,55,58,61,69,78,88,97,105,115,120,127]
refined_idx_2 = []

for ii,i in enumerate(refined_idx_1):
    if not ii in to_exclude:
        refined_idx_2.append(i)


traj_file = open("../outputs/find_traj_idx.xyz", "w")
for ii,i in enumerate(refined_idx_2):

    traj_file.write(str(len(xyz[i])))
    traj_file.write("\n\n")

    for atm in range(len(xyz[i])):
        traj_file.write(nuc_char[zs[i, atm]] + "\t")
        for coord in range(3):
            traj_file.write(str(xyz[i][atm][coord])+"\t")
        traj_file.write("\n")


traj_file.close()

final_traj_idx = np.asarray(refined_idx_2)

np.savez("../outputs/find_traj_idx.npz", np.asarray(final_traj_idx))

# final_traj_idx = np.asarray(final_traj_idx)
x = np.asarray(range(len(final_traj_idx)))*0.5
ref_ene = -133.2 * 2625.5

fig, ax = plt.subplots(figsize=(10,7))
ax.scatter(x, ene[final_traj_idx]-ref_ene, s=80)
ax.set_xlabel("Time (fs)")
ax.set_ylabel("Relative Energy (kJ/mol)")
plt.tight_layout()
plt.savefig("../plots/find_traj_idx.png", dpi=200)
plt.show()

