
# Log10 (Lattice thermal conductivity @ 300K - W/m/K) - log_klat_300

## ARFS Top features

### ARFS selected descriptors

![Relevant Descriptors](../_static/log_klat_300/arfs_feat_imp_log_klat_300.png)

---

## Correlation analysis

### Distance correlation

![Distance correlation heatmap](../_static/log_klat_300/dcor_heatmap.png)

### Dependency graphs

![Dependency graph](../_static/log_klat_300/log_klat_300_m2l_learnability.png)

### Feature learnability

![Feature learnability](../_static/log_klat_300/log_klat_300_feat_metrics.png)

---

## Model performance

### Metrics overview

RF - MATMINER

|      |   train_rmse |   test_rmse |   train_errors |   test_errors |    train_r2 |   test_r2 |
|:-----|-------------:|------------:|---------------:|--------------:|------------:|----------:|
| mean |   0.1026     |   0.27926   |    0.07024     |    0.19148    | 0.97528     | 0.81652   |
| min  |   0.101      |   0.2566    |    0.0693      |    0.1841     | 0.9746      | 0.8009    |
| max  |   0.1041     |   0.2923    |    0.0712      |    0.2        | 0.9761      | 0.8326    |
| std  |   0.00120333 |   0.0133804 |    0.000714423 |    0.00597709 | 0.000541849 | 0.0130765 |

RF - MATMINER+LOBSTER

|      |   train_rmse |   test_rmse |   train_errors |   test_errors |    train_r2 |   test_r2 |
|:-----|-------------:|------------:|---------------:|--------------:|------------:|----------:|
| mean |   0.1008     |   0.27444   |    0.06908     |    0.18766    | 0.97614     | 0.82278   |
| min  |   0.099      |   0.2533    |    0.0677      |    0.1809     | 0.9755      | 0.8081    |
| max  |   0.1028     |   0.2857    |    0.0702      |    0.1974     | 0.977       | 0.8379    |
| std  |   0.00132514 |   0.0128349 |    0.000884081 |    0.00598184 | 0.000523832 | 0.0124532 |

MODNet - MATMINER

|      |   train_rmse |   test_rmse |   train_errors |   test_errors |   train_r2 |   test_r2 |
|:-----|-------------:|------------:|---------------:|--------------:|-----------:|----------:|
| mean |   0.08274    |   0.2476    |      0.0453    |    0.15712    | 0.9839     | 0.85544   |
| min  |   0.0798     |   0.2192    |      0.0439    |    0.1515     | 0.9817     | 0.8279    |
| max  |   0.0883     |   0.2718    |      0.0477    |    0.1672     | 0.9849     | 0.8778    |
| std  |   0.00321098 |   0.0187861 |      0.0013755 |    0.00596101 | 0.00118828 | 0.0186381 |

MODNet - MATMINER+LOBSTER

|      |   train_rmse |   test_rmse |   train_errors |   test_errors |   train_r2 |   test_r2 |
|:-----|-------------:|------------:|---------------:|--------------:|-----------:|----------:|
| mean |     0.10988  |   0.247     |      0.05838   |    0.15492    | 0.97054    | 0.8559    |
| min  |     0.0696   |   0.2193    |      0.0379    |    0.1444     | 0.9614     | 0.8316    |
| max  |     0.1275   |   0.2689    |      0.0682    |    0.1668     | 0.9886     | 0.8854    |
| std  |     0.021283 |   0.0212158 |      0.0119752 |    0.00912697 | 0.00989962 | 0.0214633 |

---

## Model Explainer

### PFI
![RF pfi](../_static/log_klat_300/rf_pfi_log_klat_300.png)
![MODNet pfi](../_static/log_klat_300/modnet_pfi_log_klat_300.png)

### SHAP
![RF shap](../_static/log_klat_300/rf_shap_log_klat_300.png)
![MODNet Shap](../_static/log_klat_300/modnet_shap_log_klat_300.png)

---

## SISSO Models

### Rung 1
\begin{align*}
& log\_klat\_300 = c_0 \\
    & + a_0\left(\frac{ OxidationStates_{std\_ dev\_ oxidation\_ state} }{ DensityFeatures_{vpa} } \right) \\
    & + a_1\left(\sqrt[3]{ asi_{sum} }\right)
\end{align*}

### Rung 2
\begin{align*}
& log\_klat\_300 = c_0 \\
    & + a_0\left(\left(pair_{bwdf\_ kurtosis\_ mean} OPSiteFingerprint_{mean\_ tetrahedral\_ CN\_ 4}\right) + \\ \left(EIN_{ICOHP} OxidationStates_{std\_ dev\_ oxidation\_ state}\right)\right) \\
    & + a_1\left(\frac{ \left(\sqrt[3]{ GaussianSymmFunc_{mean\_ G4\_ 0.005\_ 1.0\_ 1.0} }\right) }{ \left(RadialDistributionFunction_{rdf\_ \_ 13.70000\_ -\_ 13.80000\_ A} + ElementProperty_{MagpieData\_ maximum\_ MendeleevNumber}\right) } \right)
\end{align*}

---

## Misc

### ARFS n-iter convergence checks

![Convergence](../_static/log_klat_300/log_klat_300_matminer_lob_n_iter_convergence.png)

