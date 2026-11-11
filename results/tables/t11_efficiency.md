**Computational cost on one Tesla V100-SXM2-16GB (batch inference, mixed precision).**

| Model                    |   Params (M) |   Fine-tune 5% (min) |   Inference (ms/img) | SSL pretraining (GPU-h)   |
|:-------------------------|-------------:|---------------------:|---------------------:|:--------------------------|
| ResNet-50                |         23.5 |                 10   |                 7    | 0 (public weights)        |
| Inception-v3             |         21.8 |                 12.4 |                11.65 | 0 (public weights)        |
| EfficientNet-B0          |          4   |                  6.2 |                 2.07 | 0 (public weights)        |
| ConvNeXt-T               |         27.8 |                  3.6 |                 1.63 | 0 (public weights)        |
| ViT-S/16 (sup. IN)       |         21.7 |                  1.8 |                 1.18 | 0 (public weights)        |
| DINO ViT-S/16 (IN)       |         21.7 |                  1.8 |                 1.19 | 0 (public weights)        |
| DINO-generic (SEM)       |         21.7 |                  1.8 |                 1.19 | 1.72                      |
| Ours: DINO-physics (SEM) |         21.7 |                  1.8 |                 1.19 | 1.80                      |
