# Order of execution

Following first two scripts should executed in the conda environment obtained using `abipy_env.yml`. This scripts will generate the necessary data needed to execute scripts numbered 3 and 4.

1. `convert_ddb_to_phonopy.ipynb`: This script generates required phonopy FORCE_CONSTANTS, BORN, POSCAR and supercell_POSCAR file using input ddb file.
2. `msd_convergence.ipynb`: This script will run phonopy and extract mean squared displacement data for different mesh sizes, temperatures and mesh sizes.

All the following scripts should be executed in the `ml_env` environment used for ML model training. See the `README.md` at the root of this repository to create this environment.

3. `extract_max_pfc.ipynb`: Script will extract and save maximum of projected force constant dataset.
4. `extract_converged_msd.ipynb`: Uses the output of `msd_convergence.ipynb` script to extract the MSD dataset.
5. `extract_H_S_U_Cv.ipynb`: Saves the H,S,U,Cv dataset for 25, 305 and 705K.
6. `extract_last_ph_dos_peak.ipynb`: Extracts and saves the last phonon dos peak dataset.
