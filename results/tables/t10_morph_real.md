**Morphology extraction on the 534 real NFFA 'Particles' test images; physical scale read automatically from the info bar for 398 images.**

| Method               | Images with ≥5 particles   |   Particles / image (median) |   Median diameter (nm), median over images |   Circularity (median) |
|:---------------------|:---------------------------|-----------------------------:|-------------------------------------------:|-----------------------:|
| Otsu + watershed     | 486/534                    |                           73 |                                         41 |                   0.99 |
| Decoder on DINO (IN) | 385/534                    |                           15 |                                         51 |                   0.85 |
| Decoder on ours      | 329/534                    |                            9 |                                         52 |                   0.86 |
