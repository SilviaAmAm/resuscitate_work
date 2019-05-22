import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style("white")
sns.set_context("poster")

files = glob.glob("/Volumes/Transcend/data_sets/dftb/*.xyz")

energies_kj = []
cc_distance = []

for f in files:
    f_in = open(f, "r")
    lines = f_in.read()
    lines = lines.split("\n")
    idx_start = lines[1].find("Energy: ") + len("Energy: ")
    idx_end = lines[1].find("(hartree)") -1
    ene_ha = float(lines[1][idx_start:idx_end])
    energies_kj.append(ene_ha * 2625.5)

    c1 = lines[2].split("\t")[1:]
    c2 = lines[7].split("\t")[1:]
    dist = 0
    for i in range(len(c1)):
        dist += (float(c1[i]) - float(c2[i]))**2
    cc_distance.append(np.sqrt(dist))

idx_sorted = np.argsort(cc_distance)
energies_kj = np.asarray(energies_kj)[idx_sorted]
cc_distance = np.asarray(cc_distance)[idx_sorted]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(cc_distance, energies_kj, alpha=0.4, s=20, label='DFTB')
ax.set_xlabel("CC distance (Angstroms)")
ax.set_ylabel("Energy (kJ/mol)")
plt.tight_layout()
plt.savefig("../plots/dftb_data.png", dpi=200)
plt.show()
