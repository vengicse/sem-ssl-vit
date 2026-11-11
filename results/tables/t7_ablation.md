**Augmentation ablation (6-epoch DINO pretraining, frozen features): k-NN and linear-probe accuracy (%), linear-probe mean corruption accuracy on SEM-C, and accuracy on the artefact removed from pretraining (leave-one-operator-out).**

| Pretraining augmentation       | k-NN 1%          | Linear 1%        | Linear 10%       | Linear mCA 10%   |   Acc. on removed artefact |
|:-------------------------------|:-----------------|:-----------------|:-----------------|:-----------------|---------------------------:|
| Generic augmentation only      | 78.67 ± 0.60     | 70.22 ± 1.88     | 78.38 ± 0.19     | 73.28            |                     nan    |
| Full physics chain             | 80.13 ± 0.95     | 73.27 ± 1.05     | 79.73 ± 0.06     | 78.31            |                     nan    |
| Physics chain without drift    | 80.17 ± 1.02     | 72.98 ± 1.24     | 80.04 ± 0.17     | 78.46            |                      78.58 |
| Physics chain without psf      | 80.02 ± 0.52     | 73.12 ± 1.16     | **80.08 ± 0.19** | 78.40            |                      77.15 |
| Physics chain without charging | **80.31 ± 0.89** | 73.33 ± 1.80     | 79.94 ± 0.34     | 77.55            |                      74.29 |
| Physics chain without gain     | 80.12 ± 0.85     | **73.36 ± 0.99** | 79.92 ± 0.19     | **78.57**        |                      79.26 |
| Physics chain without noise    | 79.08 ± 0.30     | 71.81 ± 1.39     | 78.93 ± 0.19     | 75.40            |                      64.21 |
