import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set_context("poster")
sns.set_style("white")

data_acsf = np.load("../outputs/ch4cn_acsf_002.npz")
acsf_ene_pred, acsf_ene_true, acsf_ene_pred_traj, acsf_ene_traj_true = data_acsf["arr_0"], data_acsf["arr_1"], data_acsf["arr_2"], data_acsf["arr_3"]

x = np.asarray(range(len(acsf_ene_traj_true)))*0.5
fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(x, acsf_ene_pred_traj, s=40, label="Prediction")
ax.scatter(x, acsf_ene_traj_true, s=40, label="True")
ax.legend()
ax.set_xlabel("Time (fs)")
ax.set_ylabel("Energy (kJ/mol)")
plt.tight_layout()
plt.savefig("ch4cn_acsf_002_traj.png", dpi=150)

fog, ex = plt.subplots(figsize=(8,6))
ex.scatter(acsf_ene_true, acsf_ene_pred, alpha=0.5, s=40, label="ACSF")
ex.plot([min(acsf_ene_true), max(acsf_ene_true)], [min(acsf_ene_true), max(acsf_ene_true)], c="black")
ex.set_xlabel('True energies (kJ/mol)')
ex.set_ylabel('Predicted energies (kJ/mol)')
plt.tight_layout()
plt.savefig("ch4cn_acsf_002_corr.png", dpi=150)