# sem-ssl-vit

Code, results and pretrained weights for the article

**Label-efficient self-supervised vision transformers with physics-aware augmentation for automated SEM morphology characterization of nanostructures**

K. Vengatesan, M. Ramkumar, R. Manikandan, P. Anandan, Jaehyuk Cho

A ViT-S/16 is pretrained by DINO self-distillation on unlabelled scanning electron microscopy (SEM) images. The views come from a physics-aware forward model of the SEM imaging chain: scan drift with line jitter, an astigmatic point-spread function, a charging field with scan streaks, detector gain, and Poisson–Gaussian noise. The pretrained backbone is evaluated for label-efficient classification (1/5/10/100% labels) on the NFFA-EUROPE 100% SEM Dataset, for robustness on the SEM-C corruption benchmark, and for particle morphology extraction.

## Main results (test split, 3,147 images)

| Model | Acc. 1% labels | Acc. 10% labels | Acc. 100% labels | SEM-C mCA (10% labels) |
|:--|--:|--:|--:|--:|
| ResNet-50 | 76.93 | 86.69 | – | 80.26 |
| ViT-S/16 (supervised ImageNet) | 78.87 | 88.47 | 94.09 | 83.00 |
| DINO ViT-S/16 (ImageNet) | 80.04 | 89.45 | 95.52 | 84.40 |
| DINO-generic (SEM) | 82.97 | 90.69 | 95.55 | 84.26 |
| **Ours: DINO-physics (SEM)** | **83.13** | 90.34 | 94.98 | **88.98** |

All numbers: [`results/RESULTS.md`](results/RESULTS.md) and [`results/numbers.json`](results/numbers.json). The secondary numbers quoted in the article (McNemar tests, per-class recall, per-severity SEM-C accuracy, frozen-probe robustness, real-image agreement) are derived from the stored outputs by [`verification/derive_numbers.py`](verification/derive_numbers.py) into [`verification/derived.json`](verification/derived.json) and [`verification/real_agreement.json`](verification/real_agreement.json).

## Repository layout

| Path | Contents |
|:--|:--|
| `run_all.py` | Pipeline stages: `data`, `pretrain`, `frozen`, `finetune`, `morph`, `xai`, `report` |
| `src/config.py` | Every setting used in the article (budgets, seeds, schedules, operator ranges) |
| `src/prepare_data.py` | Info-bar removal, dHash de-duplication, group-aware 70/15/15 split, nested label budgets (fixed seeds) |
| `src/physics.py` | The five GPU-batched SEM operators and the forward-model chain |
| `src/dino.py` | DINO pretraining (ViT-S/16, multi-crop, EMA teacher) and feature extraction |
| `src/train.py` | Fine-tuning (layer-wise LR decay, label smoothing), k-NN and linear probes, SEM-C |
| `src/morph.py`, `src/scalebar.py` | Synthetic particle renderer, segmentation decoder, watershed, scale-bar OCR |
| `src/metrics.py`, `src/report.py` | Metrics, statistics (paired tests, Holm, McNemar), tables and figures |
| `scripts/` | RunPod launch (`pod_run.sh`, `pod_resume.sh`), self-stop and transfer helpers |
| `results/` | Raw per-run outputs: `finetune/` (test-set probabilities), `frozen/`, `pretrain/` logs, `morph/`, `xai/`; plus `tables/` and `figures/` |
| `ckpt/dino_physics/teacher_backbone.pt` | Pretrained physics-aware ViT-S/16 backbone (timm `vit_small_patch16_224`) |
| `logs/run.log` | Full log of the GPU run (Tesla V100-SXM2-16GB) |
| `verification/` | Re-derivation of every secondary number quoted in the article |

## Reproducing

```bash
pip install -r requirements.txt          # plus a CUDA build of torch/torchvision
bash scripts/pod_run.sh                  # full pipeline on one GPU (PROFILE=low, as in the article)
python run_all.py --stages report        # regenerate results/tables, figures and RESULTS.md (CPU)
python verification/derive_numbers.py results verification/derived.json
```

The data stage downloads the NFFA-EUROPE 100% SEM Dataset (Aversa et al., *Sci. Data* 5, 180172, 2018; CC BY 4.0, [doi:10.23728/b2share.80df8606fcdb4b2bae1656f0dc6db8ba](https://doi.org/10.23728/b2share.80df8606fcdb4b2bae1656f0dc6db8ba)). The images are not redistributed here. Split and label-budget indices are regenerated deterministically from fixed seeds. `results/morph/real_examples.npz` contains a few crops of NFFA test images, used under the dataset's CC BY 4.0 licence.

Loading the pretrained backbone:

```python
import timm, torch
model = timm.create_model("vit_small_patch16_224", num_classes=0)
model.load_state_dict(torch.load("ckpt/dino_physics/teacher_backbone.pt", map_location="cpu"))
```

## Compute

One rented NVIDIA Tesla V100-SXM2 (16 GB). Main pretraining: 1.80 GPU-h (physics-aware) and 1.72 GPU-h (generic). All 34 fine-tuning runs: 1.89 GPU-h. Total GPU cost of the study: US$3.36.
