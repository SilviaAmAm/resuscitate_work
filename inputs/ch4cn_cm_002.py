import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set_context("poster")
sns.set_style("white")

data = np.load("../outputs/ch4cn_cm_001.npz")
ene_pred, ene_true, ene_pred_traj, ene_traj_true = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"]

x = np.asarray(range(len(ene_pred_traj)))*0.5
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(x, ene_traj_true, label="CCSD")
ax.scatter(x, ene_pred_traj, label="Predictions")
ax.set_xlabel("Time (fs)")
ax.set_ylabel("Energy (kJ/mol)")
plt.legend()
plt.savefig("../plots/ch4cn_cm_002_traj.png", dpi=200)

fog, ex = plt.subplots(figsize=(12,8))
ex.scatter(ene_true, ene_pred, alpha=0.5)
ex.plot([min(ene_true), max(ene_true)], [min(ene_true), max(ene_true)], c="black")
ex.set_xlabel("True energies (kJ/mol)")
ex.set_ylabel("Predicted energies (kJ/mol)")
plt.savefig("../plots/ch4cn_cm_002_corr.png", dpi=200)
