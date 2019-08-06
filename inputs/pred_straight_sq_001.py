import numpy as np
from qml.aglaia.aglaia import ARMP

# Reading the coordinates
f_squalane = open("../datasets/straight_squalane.xyz", "r")

xyz_sq = []
zs_sq = []
dic_sq = {"C": 6, "N":7, "H": 1}

for i, line in enumerate(f_squalane):
    if i == 0 or i == 1:
        continue
    
    line_split = line.rstrip().split(",")
    zs_sq.append(dic_sq[line_split[0]])
    xyz = []
    for j in range(1,4):
        xyz.append(float(line_split[j]))

    xyz_sq.append(xyz)


xyz_sq = np.asarray([xyz_sq])
zs_sq = np.asarray([zs_sq])
ene_sq = np.asarray([0])

acsf_hyperparameters = {"n_basis":16, "r_min": 0.8, "r_cut": 3.0959454963762645, "tau": 1.7612032005732925}
n_basis = acsf_hyperparameters["n_basis"] 
r_min = acsf_hyperparameters["r_min"]
r_cut = acsf_hyperparameters["r_cut"]
tau = acsf_hyperparameters["tau"]
eta = 4 * np.log(tau) * ((n_basis-1)/(r_cut - r_min))**2
zeta = - np.log(tau) / (2*np.log(np.cos(np.pi/(4*n_basis-4))))
acsf_params={"nRs2":n_basis, "nRs3":n_basis, "nTs":n_basis, "rcut":r_cut, "acut":r_cut, "zeta":zeta, "eta":eta}

estimator = ARMP(representation_name='acsf', representation_params=acsf_params)
estimator.load_nn("../tmp_models/hc6_sq")
estimator.set_properties(ene_sq)
estimator.generate_representation(xyz_sq, zs_sq, method='fortran')
ene_NN = estimator.predict([0])

print(ene_NN)

