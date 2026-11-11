**Statistical comparison of the proposed model with each existing model over all matched (budget × seed) fine-tuning runs available for both models (count in the Wins column): mean accuracy difference, wins, paired t-test, Wilcoxon signed-rank test, Holm-adjusted p (t-test), effect size d_z, and the number of seeds where the exact McNemar test on test-set predictions gives p < 0.05.**

| Ours vs            |   Δ acc (pp) | Wins   |   Paired t p |   Wilcoxon p |   d_z | McNemar p<0.05 @1%   | McNemar p<0.05 @10%   |   Holm p |
|:-------------------|-------------:|:-------|-------------:|-------------:|------:|:---------------------|:----------------------|---------:|
| ResNet-50          |         4.53 | 3/3    |       0.015  |        0.25  |  4.71 | 1/1                  | 1/1                   |    0.088 |
| Inception-v3       |         7.9  | 2/2    |       0.15   |        0.5   |  3    | 1/1                  | –                     |    0.44  |
| EfficientNet-B0    |        11.33 | 2/2    |       0.15   |        0.5   |  2.85 | 1/1                  | –                     |    0.44  |
| ConvNeXt-T         |         2.86 | 2/2    |       0.042  |        0.5   | 10.61 | 1/1                  | –                     |    0.21  |
| ViT-S/16 (sup. IN) |         2.52 | 6/6    |       0.0082 |        0.031 |  1.73 | 2/2                  | 1/1                   |    0.057 |
| DINO ViT-S/16 (IN) |         1.32 | 5/6    |       0.082  |        0.094 |  0.89 | 2/2                  | 1/1                   |    0.33  |
| DINO-generic (SEM) |        -0.2  | 1/6    |       0.43   |        0.44  | -0.35 | 0/2                  | 0/1                   |    0.44  |
