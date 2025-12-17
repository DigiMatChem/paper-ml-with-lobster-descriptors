
# Max projected force constant - eV/Å² - max_pfc

## ARFS Top features

### ARFS selected descriptors

![Relevant Descriptors](../_static/max_pfc/arfs_feat_imp_max_pfc.png)

---

## Correlation analysis

### Distance correlation

![Distance correlation heatmap](../_static/max_pfc/dcor_heatmap.png)

### Dependency graphs

![Dependency graph](../_static/max_pfc/max_pfc_m2l_learnability.png)

### Feature learnability

![Feature learnability](../_static/max_pfc/max_pfc_feat_metrics.png)

---

## Model performance

### Metrics overview

RF - MATMINER

|      |   train_rmse |   test_rmse |   train_errors |   test_errors |    train_r2 |   test_r2 |
|:-----|-------------:|------------:|---------------:|--------------:|------------:|----------:|
| mean |    1.2681    |     3.27856 |       0.6098   |      1.58038  | 0.98746     | 0.91502   |
| min  |    1.2202    |     2.7068  |       0.5841   |      1.398    | 0.986       | 0.893     |
| max  |    1.3047    |     3.8461  |       0.6404   |      1.8821   | 0.9884      | 0.9344    |
| std  |    0.0345084 |     0.40731 |       0.021375 |      0.167934 | 0.000877724 | 0.0163886 |

RF - MATMINER+LOBSTER

|      |   train_rmse |   test_rmse |   train_errors |   test_errors |    train_r2 |   test_r2 |
|:-----|-------------:|------------:|---------------:|--------------:|------------:|----------:|
| mean |    1.0706    |    2.88512  |      0.52972   |      1.41622  | 0.99104     | 0.93462   |
| min  |    1.0324    |    2.1043   |      0.5039    |      1.1474   | 0.9897      | 0.9098    |
| max  |    1.1141    |    3.8749   |      0.5558    |      1.6568   | 0.9917      | 0.9596    |
| std  |    0.0331545 |    0.598448 |      0.0189623 |      0.177869 | 0.000757892 | 0.0176389 |

MODNet - MATMINER

|      |   train_rmse |   test_rmse |   train_errors |   test_errors |    train_r2 |   test_r2 |
|:-----|-------------:|------------:|---------------:|--------------:|------------:|----------:|
| mean |    0.80204   |    2.69718  |      0.37978   |      1.21098  | 0.99496     |  0.94278  |
| min  |    0.7185    |    1.9492   |      0.3462    |      0.9539   | 0.9938      |  0.9344   |
| max  |    0.8622    |    3.2805   |      0.4232    |      1.3552   | 0.9961      |  0.9653   |
| std  |    0.0674837 |    0.429928 |      0.0267723 |      0.144313 | 0.000958332 |  0.011443 |

MODNet - MATMINER+LOBSTER

|      |   train_rmse |   test_rmse |   train_errors |   test_errors |    train_r2 |   test_r2 |
|:-----|-------------:|------------:|---------------:|--------------:|------------:|----------:|
| mean |    0.61622   |    2.27172  |      0.33      |      1.07056  | 0.997       | 0.95922   |
| min  |    0.5156    |    1.5601   |      0.2776    |      0.8488   | 0.996       | 0.9455    |
| max  |    0.6918    |    3.0118   |      0.3801    |      1.2497   | 0.998       | 0.9778    |
| std  |    0.0684287 |    0.501724 |      0.0360137 |      0.164953 | 0.000723878 | 0.0125001 |

---

## Model Explainer

### PFI
![RF pfi](../_static/max_pfc/rf_pfi_max_pfc.png)
![MODNet pfi](../_static/max_pfc/modnet_pfi_max_pfc.png)

### SHAP
![RF shap](../_static/max_pfc/rf_shap_max_pfc.png)
![MODNet Shap](../_static/max_pfc/modnet_shap_max_pfc.png)


---

## SISSO Models

### Rung 1
\begin{align*}
& max\_pfc = c_0 \\
    & + a_0\left(\frac{ bwdf_{at\_ dist0} }{ dist_{at\_ neg\_ bwdf0} } \right) \\
    & + a_1\left(BondOrientationParameter_{std\_ dev\_ BOOP\_ Q\_ l\_ 2}  \cdot asi_{max}\right)
\end{align*}

### Rung 2
\begin{align*}
& max\_pfc = c_0 \\
    & + a_0\left(\left(\left|OxidationStates_{range\_ oxidation\_ state} - ElementProperty_{MagpieData\_ range\_ NpUnfilled}\right|\right) \\ \cdot \left(BondOrientationParameter_{std\_ dev\_ BOOP\_ Q\_ l\_ 2} + OPSiteFingerprint_{mean\_ sgl\_ bd\_ CN\_ 1}\right)\right) \\
    & + a_1\left(\frac{ \left(\frac{ bwdf_{at\_ dist0} }{ dist_{at\_ neg\_ bwdf0} } \right) }{ \left(CrystalNNFingerprint_{std\_ dev\_ wt\_ CN\_ 2} - ElementProperty_{MagpieData\_ mean\_ NsValence}\right) } \right)
\end{align*}

---

## Misc

### ARFS n-iter convergence checks

![Convergence](../_static/max_pfc/max_pfc_matminer_lob_n_iter_convergence.png)

