**SEM-C robustness benchmark: accuracy (%) under each artefact averaged over severities 1–5 (severity 5 exceeds the training range), mean corruption accuracy (mCA) and relative drop from clean accuracy, for models fine-tuned with 10% of the labels (seeds as in Table 3).**

| Model                    | Labels   | Clean        | Drift        | Psf          | Charging     | Gain         | Noise        | mCA          | Rel. drop   |
|:-------------------------|:---------|:-------------|:-------------|:-------------|:-------------|:-------------|:-------------|:-------------|:------------|
| ResNet-50                | 10%      | 86.69        | 81.82        | 82.57        | 83.82        | 86.26        | 66.83        | 80.26        | 7.41        |
| ViT-S/16 (sup. IN)       | 10%      | 88.47 ± 0.18 | 86.31 ± 0.09 | 86.73 ± 0.39 | 83.02 ± 0.11 | 88.00 ± 0.13 | 70.92 ± 2.56 | 83.00 ± 0.37 | 6.18 ± 0.61 |
| DINO ViT-S/16 (IN)       | 10%      | 89.45        | 87.09        | 85.48        | 86.74        | 89.36        | 73.33        | 84.40        | 5.64        |
| DINO-generic (SEM)       | 10%      | 90.69        | 88.90        | 86.17        | 86.77        | 90.55        | 68.88        | 84.26        | 7.09        |
| Ours: DINO-physics (SEM) | 10%      | 90.34        | 89.95        | 88.95        | 89.46        | 90.04        | 86.50        | **88.98**    | **1.51**    |
