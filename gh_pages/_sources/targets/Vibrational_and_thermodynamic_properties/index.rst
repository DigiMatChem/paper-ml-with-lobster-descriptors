Vibrational and thermodynamic properties
=========================================

Scripts to compute, extract and save vibrational and thermodynamic properties datasets

.. important::

   Numbers indicate the order in which the scripts should be executed.
   The first two scripts should be executed in the conda environment obtained using ``abipy_env.yml``.
   These scripts will generate the necessary data needed to execute scripts numbered 3 and 4.

   1. :doc:`convert_ddb_to_phonopy`

      - This script generates required phonopy FORCE_CONSTANTS, BORN, POSCAR and supercell_POSCAR file using input ddb file.
   2. :doc:`msd_convergence`

      - This script will run phonopy and extract mean squared displacement data for different mesh sizes, temperatures and mesh sizes.

   All the following scripts should be executed in the ``ml_env`` environment used for ML model training. See the :ref:`README <readme-page>` to create this environment.

   3. :doc:`extract_max_pfc`

      - Script will extract and save maximum of projected force constant dataset.
   4. :doc:`extract_converged_msd`

      - Uses the output of :doc:`msd_convergence` script to extract the MSD dataset.
   5. :doc:`extract_H_S_U_Cv`

      - Saves the H,S,U,Cv dataset for 25, 305 and 705K.
   6. :doc:`extract_last_ph_dos_peak`

      - Extracts and saves the last phonon dos peak dataset.


.. toctree::
   :maxdepth: 2

   convert_ddb_to_phonopy
   msd_convergence
   extract_max_pfc
   extract_converged_msd
   extract_H_S_U_Cv
   extract_last_ph_dos_peak
