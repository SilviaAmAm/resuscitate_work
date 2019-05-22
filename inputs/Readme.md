[analyse_dftb.py](./analyse_dftb.py): extracts some of the dftb data that could be found and analyses the minima to understand what they are.

[analyse_b3lyp.py](./analyse_b3lyp.py): extracts some of the B3LYP data (which appears to have been labelled 'CC' in the past) and understands what the 2 levels of structures are.

[surface_dftb.py](./surface_dftb.py): plots the DFTB surface as a scatter plot. The energies are plotted as a function of the CC distance.

[surface_dft_b3lyp_cc.py](./surface_dft_b3lyp_cc.py): plots the PBE, b3lyp and CCSD surfaces.

[extract_ccsd.py](./extract_ccsd.py): extracts the CCSD data (which appears to have been labelled 'B3LYP') and turns it into a hdf5 file.

[find_traj_idx.py](./find_traj_idx.py): This was used to find an abstraction trajectory among the CCSD data. It then saves the indices found to a file [find_traj_idx.npz](../outputs/find_traj_idx.npz) and plots the energy of the trajectory.

[ch4cn_ho_001.py](./ch4cn_ho_001.py): This generates the model to be optimised with Osprey for the CH4 CN reaction with the CCSD energies. The reaction to be tested on is kept separate from the data on which the cross-validation is done.

