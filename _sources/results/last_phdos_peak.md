
# Last phonon dos peak - cm-1 - last_phdos_peak

## ARFS Top features

### ARFS selected descriptors

![Relevant Descriptors](../_static/last_phdos_peak/arfs_feat_imp_last_phdos_peak.png)

---

## Correlation analysis

### Distance correlation

![Distance correlation heatmap](../_static/last_phdos_peak/dcor_heatmap.png)

### Dependency graphs

![Dependency graph](../_static/last_phdos_peak/last_phdos_peak_feat_metrics.png)

### Feature learnability

![Feature learnability](../_static/last_phdos_peak/last_phdos_peak_m2l_learnability.png)

---

## Model performance

### 5-Fold CV Metrics overview

**RF - MATMINER**

|      |   train_rmse |   test_rmse |   train_errors |   test_errors |   train_r2 |   test_r2 |
|:-----|-------------:|------------:|---------------:|--------------:|-----------:|----------:|
| mean |     32.7937  |     84.3866 |      14.6748   |      39.5086  | 0.9956     | 0.96906   |
| min  |     30.4141  |     55.9272 |      14.2786   |      31.0462  | 0.9951     | 0.954     |
| max  |     34.5876  |    106.683  |      14.9702   |      44.2828  | 0.9961     | 0.9853    |
| std  |      1.47065 |     18.1411 |       0.251989 |       4.91545 | 0.00032249 | 0.0127694 |

**RF - MATMINER+LOBSTER**

|      |   train_rmse |   test_rmse |   train_errors |   test_errors |    train_r2 |   test_r2 |
|:-----|-------------:|------------:|---------------:|--------------:|------------:|----------:|
| mean |     32.5001  |     80.4644 |      14.6361   |      39.1976  | 0.99564     |  0.97226  |
| min  |     26.9725  |     61.1432 |      13.7767   |      33.5999  | 0.9949      |  0.9492   |
| max  |     35.1927  |    115.35   |      15.3407   |      45.2728  | 0.997       |  0.9824   |
| std  |      2.91981 |     18.3066 |       0.512474 |       4.11005 | 0.000728286 |  0.012148 |

**MODNet - MATMINER**

|      |   train_rmse |   test_rmse |   train_errors |   test_errors |    train_r2 |   test_r2 |
|:-----|-------------:|------------:|---------------:|--------------:|------------:|----------:|
| mean |     20.4586  |     67.3629 |      11.1928   |      33.3479  | 0.99828     |  0.97842  |
| min  |     18.6992  |     37.2775 |      10.5934   |      24.3813  | 0.998       |  0.9416   |
| max  |     22.5584  |    123.586  |      11.9387   |      46.2481  | 0.9985      |  0.9935   |
| std  |      1.38787 |     29.3642 |       0.475595 |       7.19291 | 0.000203961 |  0.018805 |

**MODNet - MATMINER+LOBSTER**

|      |   train_rmse |   test_rmse |   train_errors |   test_errors |    train_r2 |   test_r2 |
|:-----|-------------:|------------:|---------------:|--------------:|------------:|----------:|
| mean |     22.174   |     62.5857 |       10.7228  |       30.6388 | 0.99792     | 0.982     |
| min  |     17.8546  |     43.5458 |        9.5286  |       27.3127 | 0.9972      | 0.9533    |
| max  |     26.3936  |    110.553  |       12.5559  |       36.6207 | 0.9986      | 0.9911    |
| std  |      3.52212 |     24.3505 |        1.23771 |        3.4089 | 0.000577581 | 0.0143975 |

### Corrected resampled t-test on 10-fold CV 

**Summary**
|        |   t_stat |   p_value | significance_stars   |      d_av |   rel_improvement |   percent_folds_improved |
|:-------|---------:|----------:|:---------------------|----------:|------------------:|-------------------------:|
| RF     | 0.123344 |  0.452273 |                      | 0.0150349 |          0.317817 |                       50 |
| MODNet | 1.02389  |  0.166309 |                      | 0.26788   |          5.40627  |                       70 |

![RF t-test](../_static/last_phdos_peak/rf_t_test.png)

![MODNet t-test](../_static/last_phdos_peak/modnet_t_test.png)

---

## Model Explainer

### PFI
![RF pfi](../_static/last_phdos_peak/rf_pfi_last_phdos_peak.png)
![MODNet pfi](../_static/last_phdos_peak/modnet_pfi_last_phdos_peak.png)

### SHAP
![RF shap](../_static/last_phdos_peak/rf_shap_last_phdos_peak.png)
![MODNet Shap](../_static/last_phdos_peak/modnet_shap_last_phdos_peak.png)

---

## Misc

### ARFS n-iter convergence checks

![Convergence](../_static/last_phdos_peak/last_phdos_peak_matminer_lob_n_iter_convergence.png)

### MAE/ fold from 10-fold CV

Alternative visual summary of input data for t-test 

![RF per fold MAEs](../_static/last_phdos_peak/rf_fold_comparison.png)

![MODNet per fold MAEs](../_static/last_phdos_peak/modnet_fold_comparison.png)

