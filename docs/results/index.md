Results
==========

Overview of RF and MODNet model metrics for all targets from 5-Fold CV runs. For detailed results from each of the methods applied for each target please click on the target name.
The entries highlighted in bold indicate the target where meaningful improvement in models predictive performance is observed on inclusion of quantum-chemical bonding descriptors.

 

| **Target** | **RF<br>(Matminer)** | **RF<br>(Matminer+LOB)** | **MODNet<br>(Matminer)** | **MODNet<br>(Matminer+LOB)** |
|-------|---------------|-------------------|-------------------|-----------------------|
| {doc}`last_ph_peak <last_phdos_peak>` (cm⁻¹) | 39.509 ± 4.915 | **39.198 ± 4.110** | 33.348 ± 7.193 | **30.639 ± 3.409** |
| {doc}`max_pfc <max_pfc>` (eV/Å²) | 1.582 ± 0.177 | **1.385 ± 0.186** | 1.243 ± 0.184 | **1.057 ± 0.188** |
| {doc}`log_g_vrh <log_g_vrh>` | 0.096 ± 0.004 | 0.091 ± 0.006 | 0.067 ± 0.004 | 0.066 ± 0.004 |
| {doc}`log_k_vrh <log_k_vrh>` | 0.085 ± 0.006 | 0.079 ± 0.005 | 0.055 ± 0.005 | 0.053 ± 0.005 |
| {doc}`log_msd_all_300 <log_msd_all_300>` | 0.076 ± 0.001 | 0.075 ± 0.001 | 0.059 ± 0.001 | 0.059 ± 0.003 |
| {doc}`log_msd_all_600 <log_msd_all_600>` | 0.079 ± 0.001 | 0.078 ± 0.001 | 0.062 ± 0.001 | 0.059 ± 0.001 |
| {doc}`log_msd_mean_300 <log_msd_mean_300>` | 0.069 ± 0.001 | 0.067 ± 0.002 | 0.051 ± 0.002 | 0.053 ± 0.004 |
| {doc}`log_msd_mean_600 <log_msd_mean_600>` | 0.072 ± 0.002 | 0.072 ± 0.002 | 0.053 ± 0.004 | 0.055 ± 0.004 |
| {doc}`log_msd_max_300 <log_msd_max_300>` | 0.084 ± 0.004 | 0.083 ± 0.003 | 0.069 ± 0.004 | 0.071 ± 0.006 |
| {doc}`log_msd_max_600 <log_msd_max_600>` | 0.086 ± 0.003 | 0.086 ± 0.003 | 0.073 ± 0.003 | 0.073 ± 0.003 |
| {doc}`log_klat_300 <log_klat_300>`| 0.190 ± 0.003 | **0.185 ± 0.002** | 0.159 ± 0.006 | **0.154 ± 0.009** |
| {doc}`log_kp_300 <log_kp_300>` | 0.224 ± 0.001 | **0.216 ± 0.002** | 0.192 ± 0.008 | **0.185 ± 0.008** |
| {doc}`Cv_25  <Cv_25>` (meV/atom)| 0.005 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.004 ± 0.000 |
| {doc}`Cv_305 <Cv_305>` (meV/atom)| 0.003 ± 0.000 | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.003 ± 0.000 |
| {doc}`Cv_705 <Cv_705>` (meV/atom)| 0.001 ± 0.000 | 0.001 ± 0.000 | 0.001 ± 0.000 | 0.001 ± 0.000 |
| {doc}`H_25 <H_25>` (meV/atom)| 2.912 ± 0.287 | 2.932 ± 0.325 | 1.507 ± 0.193 | 1.505 ± 0.208 |
| {doc}`H_305 <H_305>` (meV/atom)| 6.342 ± 0.407 | 6.371 ± 0.389 | 3.282 ± 0.331 | 3.298 ± 0.350 |
| {doc}`H_705 <H_705>` (meV/atom) | 13.418 ± 0.795 | 13.447 ± 0.859 | 7.226 ± 0.719 | 7.258 ± 0.436 |
| {doc}`S_25 <S_25>` (meV/atom)| 0.003 ± 0.000 | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.003 ± 0.000 |
| {doc}`S_305 <S_305>` (meV/atom) | 0.017 ± 0.001 | 0.017 ± 0.001 | 0.011 ± 0.001 | 0.010 ± 0.001 |
| {doc}`S_705 <S_705>` (meV/atom) | 0.018 ± 0.001 | 0.018 ± 0.001 | 0.011 ± 0.001 | 0.011 ± 0.001 |
| {doc}`U_25 <U_25>` (meV/atom) | 2.865 ± 0.312 | 2.882 ± 0.332 | 1.500 ± 0.190 | 1.485 ± 0.191 |
| {doc}`U_305 <U_305>` (meV/atom) | 1.458 ± 0.393 | 1.480 ± 0.398 | 0.888 ± 0.099 | 0.896 ± 0.179 |
| {doc}`U_705 <U_705>` (meV/atom) | 0.771 ± 0.207 | 0.763 ± 0.205 | 0.635 ± 0.093 | 0.631 ± 0.166 |


```{toctree}
:maxdepth: 1
:hidden:
max_pfc
last_phdos_peak
log_g_vrh
log_k_vrh
log_klat_300
log_kp_300
log_msd_all_300
log_msd_all_600
log_msd_mean_300
log_msd_mean_600
log_msd_max_300
log_msd_max_600
Cv_25
Cv_305
Cv_705
H_25
H_305
H_705
S_25
S_305
S_705
U_25
U_305
U_705
```
