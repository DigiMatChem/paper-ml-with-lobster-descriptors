
# Log10 (Peierls lattice thermal conductivity ) @ 300K - W/m/K) - log_kp_300

## ARFS Top features

### ARFS selected descriptors

![Relevant Descriptors](../_static/log_kp_300/arfs_feat_imp_log_kp_300.png)

---

## Correlation analysis

### Distance correlation

![Distance correlation heatmap](../_static/log_kp_300/dcor_heatmap.png)

### Dependency graphs

![Dependency graph](../_static/log_kp_300/log_kp_300_m2l_learnability.png)

### Feature learnability

![Feature learnability](../_static/log_kp_300/log_kp_300_feat_metrics.png)

---

## Model performance

### Metrics overview

RF - MATMINER

|      |   train_rmse |   test_rmse |   train_errors |   test_errors |   train_r2 |   test_r2 |
|:-----|-------------:|------------:|---------------:|--------------:|-----------:|----------:|
| mean |    0.12314   |   0.33442   |     0.08226    |    0.22446    | 0.97192    | 0.79256   |
| min  |    0.1196    |   0.3116    |     0.0805     |    0.2153     | 0.9704     | 0.7776    |
| max  |    0.1264    |   0.3585    |     0.0839     |    0.2345     | 0.9734     | 0.8166    |
| std  |    0.0022905 |   0.0182519 |     0.00123386 |    0.00781245 | 0.00095373 | 0.0155551 |

RF - MATMINER+LOBSTER

|      |   train_rmse |   test_rmse |   train_errors |   test_errors |   train_r2 |   test_r2 |
|:-----|-------------:|------------:|---------------:|--------------:|-----------:|----------:|
| mean |   0.12088    |   0.32832   |     0.08078    |    0.21938    | 0.97294    | 0.80012   |
| min  |   0.1185     |   0.3089    |     0.0788     |    0.2058     | 0.9717     | 0.7809    |
| max  |   0.1236     |   0.3558    |     0.082      |    0.2277     | 0.9736     | 0.8217    |
| std  |   0.00191458 |   0.0177316 |     0.00116688 |    0.00781419 | 0.00069455 | 0.0140242 |

MODNet - MATMINER

|      |   train_rmse |   test_rmse |   train_errors |   test_errors |   train_r2 |   test_r2 |
|:-----|-------------:|------------:|---------------:|--------------:|-----------:|----------:|
| mean |    0.11622   |   0.30574   |      0.0635    |    0.192      |  0.97316   |  0.82628  |
| min  |    0.0864    |   0.2755    |      0.0506    |    0.1827     |  0.9416    |  0.7999   |
| max  |    0.1779    |   0.3292    |      0.0954    |    0.2062     |  0.9859    |  0.855    |
| std  |    0.0317698 |   0.0233441 |      0.0161813 |    0.00895321 |  0.0159871 |  0.021248 |

MODNet - MATMINER+LOBSTER

|      |   train_rmse |   test_rmse |   train_errors |   test_errors |   train_r2 |   test_r2 |
|:-----|-------------:|------------:|---------------:|--------------:|-----------:|----------:|
| mean |    0.10094   |   0.30428   |      0.0572    |    0.19034    |  0.97994   | 0.82824   |
| min  |    0.0782    |   0.2749    |      0.0471    |    0.1807     |  0.9591    | 0.8101    |
| max  |    0.1488    |   0.3244    |      0.0815    |    0.2003     |  0.9887    | 0.8482    |
| std  |    0.0251375 |   0.0189124 |      0.0128984 |    0.00762669 |  0.0107781 | 0.0153183 |

---

## Model Explainer

### PFI
![RF pfi](../_static/log_kp_300/rf_pfi_log_kp_300.png)
![MODNet pfi](../_static/log_kp_300/modnet_pfi_log_kp_300.png)

### SHAP
![RF shap](../_static/log_kp_300/rf_shap_log_kp_300.png)
![MODNet Shap](../_static/log_kp_300/modnet_shap_log_kp_300.png)

---

## SISSO Models

### Rung 1
\begin{align*}
& log\_kp\_300 = c_0 \\
    & + a_0\left(\sqrt[3]{ asi_{sum} }\right) \\
    & + a_1\left(\frac{ OxidationStates_{std\_ dev\_ oxidation\_ state} }{ DensityFeatures_{vpa} } \right)
\end{align*}

### Rung 2
\begin{align*}
& log\_kp\_300 = c_0 \\
    & + a_0\left(\left(ElementProperty_{MagpieData\_ minimum\_ Electronegativity} - CrystalNNFingerprint_{mean\_ wt\_ CN\_ 1}\right) \\ \left(\left|OxidationStates_{std\_ dev\_ oxidation\_ state} - ElementProperty_{MagpieData\_ minimum\_ Electronegativity}\right|\right)\right) \\
    & + a_1\left(\sqrt[3]{ \left(\frac{ asi_{sum} }{ GaussianSymmFunc_{mean\_ G4\_ 0.005\_ 1.0\_ 1.0} } \right) }\right)
\end{align*}

---

## Misc

### ARFS n-iter convergence checks

![Convergence](../_static/log_kp_300/log_kp_300_matminer_lob_n_iter_convergence.png)

