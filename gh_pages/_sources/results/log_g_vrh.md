
# Log10 (Shear modulus - GPa) - log_g_vrh

## ARFS Top features

### ARFS selected descriptors

![Relevant Descriptors](../_static/log_g_vrh/arfs_feat_imp_log_g_vrh.png)

---

## Correlation analysis

### Distance correlation

![Distance correlation heatmap](../_static/log_g_vrh/dcor_heatmap.png)

### Dependency graphs

![Dependency graph](../_static/log_g_vrh/log_g_vrh_feat_metrics.png)

### Feature learnability

![Feature learnability](../_static/log_g_vrh/log_g_vrh_m2l_learnability.png)

---

## Model performance

### 5-Fold CV Metrics overview

**RF - MATMINER**

|      |   train_rmse |   test_rmse |   train_errors |   test_errors |   train_r2 |   test_r2 |
|:-----|-------------:|------------:|---------------:|--------------:|-----------:|----------:|
| mean |   0.05998    |   0.16438   |    0.03464     |    0.09632    | 0.98278    |  0.869    |
| min  |   0.0521     |   0.1349    |    0.0338      |    0.0917     | 0.9804     |  0.7852   |
| max  |   0.0644     |   0.2225    |    0.0364      |    0.1036     | 0.9867     |  0.8986   |
| std  |   0.00414072 |   0.0315851 |    0.000939361 |    0.00429017 | 0.00212735 |  0.043152 |

**RF - MATMINER+LOBSTER**

|      |   train_rmse |   test_rmse |   train_errors |   test_errors |   train_r2 |   test_r2 |
|:-----|-------------:|------------:|---------------:|--------------:|-----------:|----------:|
| mean |   0.05944    |   0.15536   |     0.03328    |    0.09108    | 0.98304    | 0.88182   |
| min  |   0.0491     |   0.1206    |     0.0322     |    0.0847     | 0.9803     | 0.7944    |
| max  |   0.0646     |   0.2177    |     0.0352     |    0.0994     | 0.9882     | 0.9267    |
| std  |   0.00536865 |   0.0369465 |     0.00101666 |    0.00569048 | 0.00277604 | 0.0483531 |

**MODNet - MATMINER**

|      |   train_rmse |   test_rmse |   train_errors |   test_errors |   train_r2 |   test_r2 |
|:-----|-------------:|------------:|---------------:|--------------:|-----------:|----------:|
| mean |    0.08042   |   0.12398   |     0.02318    |    0.06704    |  0.96684   | 0.9215    |
| min  |    0.0348    |   0.0925    |     0.0202     |    0.062      |  0.9577    | 0.8215    |
| max  |    0.0947    |   0.2028    |     0.0274     |    0.0718     |  0.9941    | 0.9574    |
| std  |    0.0228767 |   0.0403218 |     0.00243343 |    0.00397522 |  0.0137634 | 0.0503536 |

**MODNet - MATMINER+LOBSTER**

|      |   train_rmse |   test_rmse |   train_errors |   test_errors |   train_r2 |   test_r2 |
|:-----|-------------:|------------:|---------------:|--------------:|-----------:|----------:|
| mean |    0.08218   |   0.1216    |     0.02634    |    0.06566    |   0.96568  | 0.92412   |
| min  |    0.0388    |   0.0897    |     0.0222     |    0.0607     |   0.9557   | 0.824     |
| max  |    0.0984    |   0.2014    |     0.0338     |    0.0707     |   0.9926   | 0.96      |
| std  |    0.0219141 |   0.0410796 |     0.00424292 |    0.00427018 |   0.013602 | 0.0505217 |

### Corrected resampled t-test on 10-fold CV

**Summary**
|        |   t_stat |     p_value | significance_stars   |     d_av |   rel_improvement |   percent_folds_improved |
|:-------|---------:|------------:|:---------------------|---------:|------------------:|-------------------------:|
| RF     |  7.56265 | 1.72926e-05 | ***                  | 0.583957 |           4.31678 |                      100 |
| MODNet |  1.01519 | 0.168263    |                      | 0.211847 |           2.51596 |                       80 |

![RF t-test](../_static/log_g_vrh/rf_t_test.png)

![MODNet t-test](../_static/log_g_vrh/modnet_t_test.png)

---

## Model Explainer

### PFI
![RF pfi](../_static/log_g_vrh/rf_pfi_log_g_vrh.png)
![MODNet pfi](../_static/log_g_vrh/modnet_pfi_log_g_vrh.png)

### SHAP
![RF shap](../_static/log_g_vrh/rf_shap_log_g_vrh.png)
![MODNet Shap](../_static/log_g_vrh/modnet_shap_log_g_vrh.png)

---

## SISSO Models

### Rung 1

#### 1D descriptor

\begin{align*}
& log\_g\_vrh = 3.086 \\
    & -0.007\left(ElementProperty_{MagpieData\_maximum\_MendeleevNumber} \\ * AverageBondLength_{mean\_Average\_bond\_length}\right)
\end{align*}

#### 2D descriptor

\begin{align*}
& log\_g\_vrh = 2.679 \\
    & + 1.127\left(MaximumPackingEfficiency_{max\_packing\_efficiency} \\ * ElementProperty_{MagpieData\_mean\_NsValence}\right) \\
    & -0.009\left(ElementProperty_{MagpieData\_maximum\_MendeleevNumber} \\ * AverageBondLength_{mean\_Average\_bond\_length}\right)
\end{align*}

### Rung 2

#### 1D descriptor

\begin{align*}
& log\_g\_vrh = 3.472 \\
    & -0.006\left(\frac{ \left(ElementProperty_{MagpieData\_maximum\_MendeleevNumber} * AverageBondLength_{mean\_Average\_bond\_length}\right) }{ \left(\sqrt[3]{ MaximumPackingEfficiency_{max\_packing\_efficiency} }\right) } \right)
\end{align*}

#### 2D descriptor

\begin{align*}
& log\_g\_vrh = 2.423 \\
    & + 0.29\left(\left(\sqrt[3]{ ElementProperty_{MagpieData\_mean\_NUnfilled} }\right) \\ + \left(\left|AGNIFingerPrint_{std\_dev\_AGNI\_dir\_y\_eta\_2.89e+00} - ElementProperty_{MagpieData\_mean\_NsValence}\right|\right)\right) \\
    & -0.006\left(\frac{ \left(ElementProperty_{MagpieData\_maximum\_MendeleevNumber} AverageBondLength_{mean\_Average\_bond\_length}\right) }{ \left(\sqrt[3]{ MaximumPackingEfficiency_{max\_packing\_efficiency} }\right) } \right)
\end{align*}

---

## Misc

### ARFS n-iter convergence checks

![Convergence](../_static/log_g_vrh/log_g_vrh_matminer_lob_n_iter_convergence.png)

### MAE/ fold from 10-fold CV

Alternative visual summary of input data for t-test

![RF per fold MAEs](../_static/log_g_vrh/rf_fold_comparison.png)

![MODNet per fold MAEs](../_static/log_g_vrh/modnet_fold_comparison.png)
