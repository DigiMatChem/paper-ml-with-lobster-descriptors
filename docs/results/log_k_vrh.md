
# Log10 (Bulk modulus - GPa) - log_k_vrh

## ARFS Top features

### ARFS selected descriptors

![Relevant Descriptors](../_static/log_k_vrh/arfs_feat_imp_log_k_vrh.png)

---

## Correlation analysis

### Distance correlation

![Distance correlation heatmap](../_static/log_k_vrh/dcor_heatmap.png)

### Dependency graphs

![Dependency graph](../_static/log_k_vrh/log_k_vrh_m2l_learnability.png)

### Feature learnability

![Feature learnability](../_static/log_k_vrh/log_k_vrh_feat_metrics.png)

---

## Model performance

### Metrics overview

RF - MATMINER

|      |   train_rmse |   test_rmse |   train_errors |   test_errors |   train_r2 |   test_r2 |
|:-----|-------------:|------------:|---------------:|--------------:|-----------:|----------:|
| mean |    0.11422   |    0.27878  |     0.03686    |    0.09826    |  0.94864   |  0.69378  |
| min  |    0.0691    |    0.1761   |     0.0333     |    0.089      |  0.9359    |  0.3988   |
| max  |    0.1323    |    0.5309   |     0.0387     |    0.112      |  0.9765    |  0.8489   |
| std  |    0.0235249 |    0.130613 |     0.00200659 |    0.00918947 |  0.0148462 |  0.158257 |

RF - MATMINER+LOBSTER

|      |   train_rmse |   test_rmse |   train_errors |   test_errors |   train_r2 |   test_r2 |
|:-----|-------------:|------------:|---------------:|--------------:|-----------:|----------:|
| mean |    0.11256   |    0.27236  |     0.03476    |    0.09474    |  0.95      |  0.70726  |
| min  |    0.065     |    0.1668   |     0.0306     |    0.0857     |  0.9385    |  0.4233   |
| max  |    0.1306    |    0.5199   |     0.0362     |    0.1054     |  0.9792    |  0.8644   |
| std  |    0.0240041 |    0.128613 |     0.00211339 |    0.00829906 |  0.0147969 |  0.153126 |

MODNet - MATMINER

|      |   train_rmse |   test_rmse |   train_errors |   test_errors |   train_r2 |   test_r2 |
|:-----|-------------:|------------:|---------------:|--------------:|-----------:|----------:|
| mean |    0.21478   |    0.21466  |     0.02622    |    0.06516    |  0.8137    |   0.80298 |
| min  |    0.1015    |    0.0954   |     0.0198     |    0.0538     |  0.7431    |   0.4391  |
| max  |    0.2634    |    0.5128   |     0.0293     |    0.0796     |  0.9494    |   0.9557  |
| std  |    0.0611977 |    0.158068 |     0.00335345 |    0.00957843 |  0.0771559 |   0.19531 |

MODNet - MATMINER+LOBSTER

|      |   train_rmse |   test_rmse |   train_errors |   test_errors |   train_r2 |   test_r2 |
|:-----|-------------:|------------:|---------------:|--------------:|-----------:|----------:|
| mean |     0.2128   |    0.21396  |     0.02756    |       0.06328 |  0.81484   |  0.79994  |
| min  |     0.1031   |    0.0961   |     0.0206     |       0.0512  |  0.7384    |  0.4384   |
| max  |     0.2669   |    0.5131   |     0.0378     |       0.0771  |  0.9477    |  0.955    |
| std  |     0.065903 |    0.162109 |     0.00657529 |       0.01073 |  0.0872064 |  0.202231 |

---

## Model Explainer

### PFI
![RF pfi](../_static/log_k_vrh/rf_pfi_log_k_vrh.png)
![MODNet pfi](../_static/log_k_vrh/modnet_pfi_log_k_vrh.png)

### SHAP
![RF shap](../_static/log_k_vrh/rf_shap_log_k_vrh.png)
![MODNet Shap](../_static/log_k_vrh/modnet_shap_log_k_vrh.png)

---

## SISSO Models

### Rung 1
\begin{align*}
& log\_k\_vrh = c_0 \\
    & + a_0\left(AGNIFingerPrint_{std\_ dev\_ AGNI\_ dir\_ z\_ eta\_ 1.88e+00} + CrystalNNFingerprint_{std\_ dev\_ wt\_ CN\_ 1}\right) \\
    & + a_1\left(\ln{ GaussianSymmFunc_{mean\_ G4\_ 0.005\_ 1.0\_ 1.0} }\right)
\end{align*}

### Rung 2
\begin{align*}
& log\_k\_vrh = c_0 \\
    & + a_0\left(\left(asi_{max} ElementProperty_{MagpieData\_ mean\_ NpValence}\right) \\ + \left(\frac{ Icohp_{sum\_ min} }{ ElementProperty_{MagpieData\_ maximum\_ MendeleevNumber} } \right)\right) \\
    & + a_1\left(\left(ElementProperty_{MagpieData\_ mean\_ CovalentRadius} ElementProperty_{MagpieData\_ mean\_ NsValence}\right) \\ \left(\ln{ GaussianSymmFunc_{mean\_ G4\_ 0.005\_ 1.0\_ 1.0} }\right)\right)
\end{align*}

---

## Misc

### ARFS n-iter convergence checks

![Convergence](../_static/log_k_vrh/log_k_vrh_matminer_lob_n_iter_convergence.png)

