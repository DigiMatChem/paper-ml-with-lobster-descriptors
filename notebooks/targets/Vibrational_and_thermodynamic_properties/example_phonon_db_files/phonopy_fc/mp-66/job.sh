#!/bin/bash
cd /home/anaik/Work/Dev_Codes/paper-ml-with-lobster-descriptors/notebooks/targets/Vibrational_and_thermodynamic_properties/example_phonon_db_files/phonopy_fc/mp-66
# OpenMp Environment
export OMP_NUM_THREADS=1
mpirun  -n 8 anaddb /home/anaik/Work/Dev_Codes/paper-ml-with-lobster-descriptors/notebooks/targets/Vibrational_and_thermodynamic_properties/example_phonon_db_files/phonopy_fc/mp-66/run.abi  > /home/anaik/Work/Dev_Codes/paper-ml-with-lobster-descriptors/notebooks/targets/Vibrational_and_thermodynamic_properties/example_phonon_db_files/phonopy_fc/mp-66/run.log 2> /home/anaik/Work/Dev_Codes/paper-ml-with-lobster-descriptors/notebooks/targets/Vibrational_and_thermodynamic_properties/example_phonon_db_files/phonopy_fc/mp-66/run.err
