
# Log10 (Mean of mean squared displacements @ 600K - Å²) - log_msd_mean_600

## ARFS Top features

### ARFS selected descriptors

![Relevant Descriptors](../_static/log_msd_mean_600/arfs_feat_imp_log_msd_mean_600.png)

---

## Correlation analysis

### Distance correlation

![Distance correlation heatmap](../_static/log_msd_mean_600/dcor_heatmap.png)

### Dependency graphs

![Dependency graph](../_static/log_msd_mean_600/log_msd_mean_600_feat_metrics.png)

### Feature learnability

![Feature learnability](../_static/log_msd_mean_600/log_msd_mean_600_m2l_learnability.png)

---

## Model performance

### 5-Fold CV Metrics overview

**RF - MATMINER**

|      |   train_rmse |   test_rmse |   train_errors |   test_errors |    train_r2 |    test_r2 |
|:-----|-------------:|------------:|---------------:|--------------:|------------:|-----------:|
| mean |  0.03622     |  0.09852    |    0.02664     |     0.07194   | 0.9865      | 0.89944    |
| min  |  0.0352      |  0.0938     |    0.0264      |     0.0689    | 0.9858      | 0.8877     |
| max  |  0.0369      |  0.103      |    0.0269      |     0.0742    | 0.9874      | 0.9098     |
| std  |  0.000584466 |  0.00346491 |    0.000162481 |     0.0023821 | 0.000517687 | 0.00826525 |

**RF - MATMINER+LOBSTER**

|      |   train_rmse |   test_rmse |   train_errors |   test_errors |    train_r2 |    test_r2 |
|:-----|-------------:|------------:|---------------:|--------------:|------------:|-----------:|
| mean |  0.0357      |  0.09858    |    0.02612     |    0.0717     | 0.98688     | 0.89952    |
| min  |  0.0348      |  0.0962     |    0.0257      |    0.0683     | 0.9865      | 0.8931     |
| max  |  0.0361      |  0.1024     |    0.0263      |    0.0741     | 0.9877      | 0.9056     |
| std  |  0.000501996 |  0.00221124 |    0.000213542 |    0.00223517 | 0.000430813 | 0.00508936 |

**MODNet - MATMINER**

|      |   train_rmse |   test_rmse |   train_errors |   test_errors |   train_r2 |   test_r2 |
|:-----|-------------:|------------:|---------------:|--------------:|-----------:|----------:|
| mean |   0.0343     |     0.07474 |     0.02326    |    0.05266    | 0.98764    | 0.94148   |
| min  |   0.031      |     0.0668  |     0.021      |    0.0485     | 0.98       | 0.9169    |
| max  |   0.0443     |     0.0877  |     0.0304     |    0.0597     | 0.99       | 0.9543    |
| std  |   0.00505134 |     0.00732 |     0.00360311 |    0.00376436 | 0.00384947 | 0.0130464 |

**MODNet - MATMINER+LOBSTER**

|      |   train_rmse |   test_rmse |   train_errors |   test_errors |   train_r2 |   test_r2 |
|:-----|-------------:|------------:|---------------:|--------------:|-----------:|----------:|
| mean |   0.03652    |  0.07694    |     0.02464    |    0.05482    | 0.98576    | 0.93834   |
| min  |   0.0294     |  0.0673     |     0.0196     |    0.0499     | 0.9756     | 0.9202    |
| max  |   0.0484     |  0.086      |     0.0317     |    0.06       | 0.9912     | 0.9512    |
| std  |   0.00690663 |  0.00684474 |     0.00435229 |    0.00384572 | 0.00563191 | 0.0110721 |

### Paired 5x2 CV F-test

**RF F-tests metrics comparsion**
![RF F-test](../_static/log_msd_mean_600/rf_f_test.png)

**MODNet F-tests metrics comparsion**
![MODNet F-test](../_static/log_msd_mean_600/modnet_f_test.png)

**Summary**
|        |   F-statistic |   p-value |      d_av |   % Relative MAE improvement | Improved folds   |
|:-------|--------------:|----------:|----------:|-----------------------------:|:-----------------|
| RF     |      1.52511  |  0.335304 | 1.17324   |                     1.2976   | 9/10             |
| MODNet |      0.761405 |  0.66715  | 0.0571183 |                     0.388444 | 6/10             |

---

## Model Explainer

### PFI
![RF pfi](../_static/log_msd_mean_600/rf_pfi_log_msd_mean_600.png)
![MODNet pfi](../_static/log_msd_mean_600/modnet_pfi_log_msd_mean_600.png)

### SHAP
![RF shap](../_static/log_msd_mean_600/rf_shap_log_msd_mean_600.png)
![MODNet Shap](../_static/log_msd_mean_600/modnet_shap_log_msd_mean_600.png)

---

## Misc

### ARFS n-iter convergence checks

![Convergence](../_static/log_msd_mean_600/log_msd_mean_600_matminer_lob_n_iter_convergence.png)

