import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set_context("poster")
sns.set_style("white")

data_cm = np.load("../outputs/ch4cn_cm_001.npz")
cm_ene_pred, cm_ene_true, cm_ene_pred_traj, cm_ene_traj_true = data_cm["arr_0"], data_cm["arr_1"], data_cm["arr_2"], data_cm["arr_3"]

data_slatm = np.load("../outputs/ch4cn_slatm_001.npz")
slatm_ene_pred, slatm_ene_true, slatm_ene_pred_traj, slatm_ene_traj_true = data_slatm["arr_0"], data_slatm["arr_1"], data_slatm["arr_2"], data_slatm["arr_3"]

data_aslatm = np.load("../outputs/ch4cn_aslatm_001.npz")
aslatm_ene_pred, aslatm_ene_true, aslatm_ene_pred_traj, aslatm_ene_traj_true = data_aslatm["arr_0"], data_aslatm["arr_1"], data_aslatm["arr_2"], data_aslatm["arr_3"]

data_acsf = np.load("../outputs/ch4cn_acsf_001.npz")
acsf_ene_pred, acsf_ene_true, acsf_ene_pred_traj, acsf_ene_traj_true = data_acsf["arr_0"], data_acsf["arr_1"], data_acsf["arr_2"], data_acsf["arr_3"]

x = np.asarray(range(len(cm_ene_pred_traj)))*0.5
fig, ax = plt.subplots(2, 2, figsize=(12,8), sharex='col', sharey='row')
ax[0,0].scatter(x, cm_ene_pred_traj, s=40)
ax[0,0].scatter(x, cm_ene_traj_true, s=40)
ax[0,0].set_title("Coulomb matrix")
ax[0,1].scatter(x, slatm_ene_pred_traj, s=40, label="SLATM")
ax[0,1].scatter(x, cm_ene_traj_true, s=40)
ax[0,1].set_title("SLATM")
ax[1,0].scatter(x, aslatm_ene_pred_traj, s=40, label="aSLATM")
ax[1,0].scatter(x, cm_ene_traj_true, s=40)
ax[1,0].set_title("aSLATM")
ax[1,1].scatter(x, acsf_ene_pred_traj, s=40, label="Prediction")
ax[1,1].scatter(x, cm_ene_traj_true, s=40, label="True")
ax[1,1].legend()
ax[1,1].set_title("ACSF")
ax[1,0].set_xlabel("Time (fs)")
ax[1,1].set_xlabel("Time (fs)")
ax[1,0].set_ylabel("Energy (kJ/mol)")
ax[0,0].set_ylabel("Energy (kJ/mol)")
plt.tight_layout()
plt.savefig("../plots/ch4cn_full_001_traj.png", dpi=200)

fog, ex = plt.subplots(2, 2, figsize=(12,8), sharex='col', sharey='row')
ex[0,0].scatter(cm_ene_true, cm_ene_pred, alpha=0.5, label="Coulomb matrix")
ex[0,0].plot([min(cm_ene_true), max(cm_ene_true)], [min(cm_ene_true), max(cm_ene_true)], c="black")
ex[0,0].set_title("Coulomb matrix")
ex[0,1].scatter(slatm_ene_true, slatm_ene_pred, alpha=0.5, label="SLATM")
ex[0,1].plot([min(slatm_ene_true), max(slatm_ene_true)], [min(slatm_ene_true), max(slatm_ene_true)], c="black")
ex[0,1].set_title("SLATM")
ex[1,0].scatter(aslatm_ene_true, aslatm_ene_pred, alpha=0.5, label="aSLATM")
ex[1,0].plot([min(aslatm_ene_true), max(aslatm_ene_true)], [min(aslatm_ene_true), max(aslatm_ene_true)], c="black")
ex[1,0].set_title("aSLATM")
ex[1,1].scatter(acsf_ene_true, acsf_ene_pred, alpha=0.5, label="ACSF")
ex[1,1].plot([min(acsf_ene_true), max(acsf_ene_true)], [min(acsf_ene_true), max(acsf_ene_true)], c="black")
ex[1,1].set_title("ACSF")
fog.text(0.5, 0.04, 'True energies (kJ/mol)', ha='center', va='center')
fog.text(0.03, 0.5, 'Predicted energies (kJ/mol)', ha='center', va='center', rotation='vertical')
plt.savefig("../plots/ch4cn_full_001_corr.png", dpi=200)
