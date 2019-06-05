# Description of inputs

[analyse_dftb.py](./analyse_dftb.py): extracts some of the dftb data that could be found and analyses the minima to understand what they are.

[analyse_b3lyp.py](./analyse_b3lyp.py): extracts some of the B3LYP data (which appears to have been labelled 'CC' in the past) and understands what the 2 levels of structures are.

[surface_dftb.py](./surface_dftb.py): plots the DFTB surface as a scatter plot. The energies are plotted as a function of the CC distance.

[surface_dft_b3lyp_cc.py](./surface_dft_b3lyp_cc.py): plots the PBE, b3lyp and CCSD surfaces.

[extract_ccsd.py](./extract_ccsd.py): extracts the CCSD data (which appears to have been labelled 'B3LYP') and turns it into a hdf5 file.

[find_traj_idx.py](./find_traj_idx.py): This was used to find an abstraction trajectory among the CCSD data. It then saves the indices found to a file [find_traj_idx.npz](../outputs/find_traj_idx.npz) and plots the energy of the trajectory.

## Chapter 1
[ch4cn_ho_001.py](./ch4cn_ho_001.py): This generates the model to be optimised with Osprey for the CH4 CN reaction with the CCSD energies and the Coulomb matrix. The reaction to be tested on is kept separate from the data on which the cross-validation is done. It produces the following files: [ch4cn_ho_001.npz](../outputs/ch4cn_ho_001.npz) with the indices of the samples used for training and testing, [ch4cn_ho_001.csv](../outputs/ch4cn_ho_001.csv) with the indices for osprey, and [ch4cn_ho_001.pickle](../outputs/ch4cn_ho_001.pickle) with the model for Osprey to use.

[ch4cn_ho_002.yaml](./ch4cn_ho_002.yaml): Used to run the osprey hyper-parameter optimisation for CH4 CN with the Coulomb matrix. It generates [ch4cn_ho_002.db](../outputs/ch4cn_ho_002.db) with the values of the hyper-parameters and [ch4cn_ho_002.out](../outputs/ch4cn_ho_002.out) with the text output of osprey.

[ch4cn_ho_003.py](./ch4cn_ho_003.py): This generates the model to be optimised with Osprey for the CH4 CN reaction with the CCSD energies and the SLATM descriptor. The reaction to be tested on is kept separate from the data on which the cross-validation is done. It produces the following files: [ch4cn_ho_003.npz](../outputs/ch4cn_ho_003.npz) with the indices of the samples used for training and testing, [ch4cn_ho_003.csv](../outputs/ch4cn_ho_003.csv) with the indices for osprey, and [ch4cn_ho_003.pickle](../outputs/ch4cn_ho_003.pickle) with the model for Osprey to use.

[ch4cn_ho_004.yaml](./ch4cn_ho_004.yaml): Used to run the osprey hyper-parameter optimisation for CH4 CN with the global SLATM descriptor. It generates [ch4cn_ho_004.db](../outputs/ch4cn_ho_004.db) with the values of the hyper-parameters and [ch4cn_ho_004.out](../outputs/ch4cn_ho_004.out) with the text output of osprey.

[ch4cn_ho_005.py](./ch4cn_ho_005.py): This generates the model to be optimised with Osprey for the CH4 CN reaction with the CCSD energies and the atomic SLATM descriptor. The reaction to be tested on is kept separate from the data on which the cross-validation is done. It produces the following files: [ch4cn_ho_005.npz](../outputs/ch4cn_ho_005.npz) with the indices of the samples used for training and testing, [ch4cn_ho_005.csv](../outputs/ch4cn_ho_005.csv) with the indices for osprey, and [ch4cn_ho_005.pickle](../outputs/ch4cn_ho_005.pickle) with the model for Osprey to use.

[ch4cn_ho_006.yaml](./ch4cn_ho_006.yaml): Used to run the osprey hyper-parameter optimisation for CH4 CN with the atomic SLATM descriptor. It generates [ch4cn_ho_006.db](../outputs/ch4cn_ho_006.db) with the values of the hyper-parameters and [ch4cn_ho_006.out](../outputs/ch4cn_ho_006.out) with the text output of osprey.

[ch4cn_ho_007.py](./ch4cn_ho_007.py): This generates the model to be optimised with Osprey for the CH4 CN reaction with the CCSD energies and the ACSFs descriptor. The reaction to be tested on is kept separate from the data on which the cross-validation is done. It produces the following files: [ch4cn_ho_007.npz](../outputs/ch4cn_ho_007.npz) with the indices of the samples used for training and testing, [ch4cn_ho_007.csv](../outputs/ch4cn_ho_007.csv) with the indices for osprey, and [ch4cn_ho_007.pickle](../outputs/ch4cn_ho_007.pickle) with the model for Osprey to use.

[ch4cn_ho_008.yaml](./ch4cn_ho_008.yaml): Used to run the osprey hyper-parameter optimisation for CH4 CN with the ACSF descriptor. It generates [ch4cn_ho_008.db](../outputs/ch4cn_ho_008.db) with the values of the hyper-parameters and [ch4cn_ho_008.out](../outputs/ch4cn_ho_008.out) with the text output of osprey.

[ch4cn_cm_001.py](./ch4cn_cm_001.py): Using the hyper-parameters found by [ch4cn_ho_002.yaml](./ch4cn_ho_002.yaml) to train a model [ch4cn_cm_001/](./ch4cn_cm_001).

[ch4cn_cm_002.py](./ch4cn_cm_002.py): plotting the results obtained in [ch4cn_cm_001.py](./ch4cn_cm_001.py). It makes 2 plots [ch4cn_cm_002_traj.png](../plots/ch4cn_cm_002_traj.png) and [ch4cn_cm_002_corr.png](../plots/ch4cn_cm_002_corr.png).

[ch4cn_acsf_001.py](./ch4cn_acsf_001.py): training the network with the ACSF and the parameters that were optimsied with [ch4cn_ho_008.yaml](./ch4cn_ho_008.yaml). It saves the trained model in [ch4cn_acsf_001/](../outputs/ch4cn_acsf_001) and saves the results of the predictions in [ch4cn_acsf_001.npz](../outputs/ch4cn_acsf_001.npz).

[ch4cn_slatm_001.py](./ch4cn_slatm_001.py): training the network with the SLATM and the parameters that were optimsied with [ch4cn_ho_004.yaml](./ch4cn_ho_004.yaml). It saves the trained model in [ch4cn_slatm_001/](../outputs/ch4cn_slatm_001) and saves the results of the predictions in [ch4cn_slatm_001.npz](../outputs/ch4cn_slatm_001.npz).

[ch4cn_aslatm_001.py](./ch4cn_aslatm_001.py): training the network with the aSLATM and the parameters that were optimsied with [ch4cn_ho_006.yaml](./ch4cn_ho_006.yaml). It saves the trained model in [ch4cn_aslatm_001/](../outputs/ch4cn_aslatm_001) and saves the results of the predictions in [ch4cn_aslatm_001.npz](../outputs/ch4cn_aslatm_001.npz).

[ch4cn_full_001.py](./ch4cn_full_001.py): plotting the results from the trained models [ch4cn_cm_001.py](./ch4cn_cm_001.py), [ch4cn_slatm_001.py](./ch4cn_slatm_001.py), [ch4cn_aslatm_001.py](./ch4cn_aslatm_001.py), and [ch4cn_acsf_001.py](./ch4cn_acsf_001.py).

[ch4cn_ho_009.py](./ch4cn_ho_009.py): This generates the model to be optimised with Osprey for the CH4 CN reaction with the CCSD energies and the ACSFs descriptor, but the parameters of the ACSF are not set (so they can be optimised). The reaction to be tested on is kept separate from the data on which the cross-validation is done. It produces the following files: [ch4cn_ho_009.npz](../outputs/ch4cn_ho_009.npz) with the indices of the samples used for training and testing, [ch4cn_ho_009.csv](../outputs/ch4cn_ho_009.csv) with the indices for osprey, and [ch4cn_ho_009.pickle](../outputs/ch4cn_ho_009.pickle) with the model for Osprey to use.

[ch4cn_ho_010.yaml](./ch4cn_ho_010.yaml): Used to run the osprey hyper-parameter optimisation for CH4 CN with the ACSF descriptor (the parameters of the ACSF are optimised). It generates [ch4cn_ho_010.db](../outputs/ch4cn_ho_010.db) with the values of the hyper-parameters and [ch4cn_ho_010.out](../outputs/ch4cn_ho_010.out) with the text output of osprey.

[ch4cn_acsf_002.py](./ch4cn_acsf_002.py): retrains the model using the ACSF and the new hyper-parameters optimised with [ch4cn_ho_010.yaml](./ch4cn_ho_010.yaml). 

[ch4cn_batchsize_001.py](./ch4cn_batchsize_001.py): runs a scan to figure out which batch size is ideal.
