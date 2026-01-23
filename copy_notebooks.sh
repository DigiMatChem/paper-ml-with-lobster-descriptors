#!/usr/bin/env bash
set -euo pipefail

# Copy ML training notebooks to documentation folder
cp -r notebooks/ml_scripts/training/*.ipynb docs/training/
cp -r notebooks/ml_scripts/correlation_analysis/*.ipynb docs/correlation_analysis/
cp -r notebooks/ml_scripts/explainer/*.ipynb docs/explainer/
cp -r notebooks/ml_scripts/feature_selector/*.ipynb docs/feature_selector/
cp -r notebooks/ml_scripts/corrected_resampled_t_test/*.ipynb docs/t_test/

# Copy miscellaneous notebooks to documentation folder
cp -r notebooks/misc/*.ipynb \
      docs/misc/

# Copy target-extraction notebooks to documentation folder
cp -r notebooks/targets/Elasticity/*.ipynb \
      docs/targets/Elasticity/

cp -r notebooks/targets/Lattice_thermal_conductivity/*.ipynb \
      docs/targets/Lattice_thermal_conductivity/

cp -r notebooks/targets/Vibrational_and_thermodynamic_properties/*.ipynb \
      docs/targets/Vibrational_and_thermodynamic_properties/
