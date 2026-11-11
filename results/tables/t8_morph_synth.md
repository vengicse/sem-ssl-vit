**Particle segmentation and size-distribution accuracy on 120 synthetic SEM test images with exact ground truth (mean ± s.d. over training seeds). Dice, F1, median-diameter and count errors in %; W1 = Wasserstein distance between predicted and true diameter distributions.**

| Method                                         | Dice             | Instance F1@0.5   | PSD W1 (px)   | Median-d error   | Count error    | F1 (s ≥ 0.75)   |
|:-----------------------------------------------|:-----------------|:------------------|:--------------|:-----------------|:---------------|:----------------|
| Otsu + watershed (classical)                   | 77.53            | 55.05             | **2.66**      | **14.24**        | **57.61**      | **30.50**       |
| CNN decoder, no pretraining, 128 labels        | 74.86 ± 0.32     | 50.94 ± 0.22      | 3.85 ± 0.12   | 24.74 ± 1.05     | 109.83 ± 19.57 | 19.31 ± 0.43    |
| Decoder on DINO ViT-S/16 (IN), 128 labels      | **78.55 ± 0.04** | 55.24 ± 0.89      | 3.74 ± 0.10   | 24.83 ± 1.04     | 58.76 ± 10.58  | 27.83 ± 0.91    |
| Decoder on ours (DINO-physics SEM), 128 labels | 77.75 ± 0.28     | **55.62 ± 1.10**  | 3.81 ± 0.18   | 25.72 ± 1.96     | 70.69 ± 22.53  | 27.63 ± 1.27    |
