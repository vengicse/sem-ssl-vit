"""Every setting quoted in the paper lives here."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SMOKE = os.environ.get("SMOKE", "0") == "1"
# PROFILE=low (default): cheapest run that still has 3 seeds, all baselines and all ablations.
# PROFILE=full: longer schedules (paper-grade upper bound, ~4x the GPU time).
LOW = os.environ.get("PROFILE", "low") == "low"
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / ("data_smoke" if SMOKE else "data")))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / ("results_smoke" if SMOKE else "results")))
CKPT_DIR = Path(os.environ.get("CKPT_DIR", ROOT / ("ckpt_smoke" if SMOKE else "ckpt")))

# ---------------------------------------------------------------- dataset
# NFFA-EUROPE "100% SEM Dataset" v2.0 (Aversa et al., Sci. Data 5, 180172, 2018), CC-BY 4.0
B2SHARE_RECORD = "zja8y-53j14"
DOI = "10.23728/b2share.80df8606fcdb4b2bae1656f0dc6db8ba"
CLASSES = ["Biological", "Fibres", "Films_Coated_Surface", "MEMS_devices_and_electrodes",
           "Nanowires", "Particles", "Patterned_surface", "Porous_Sponge", "Powder", "Tips"]
STORE_SIZE = 256          # stored resolution (grayscale, info bar removed, shorter side -> 256, centre crop)
IMG_SIZE = 224            # network input
MORPH_SIZE = 512          # stored resolution of the Particles images used for morphology
SPLIT = (0.70, 0.15, 0.15)
BUDGETS = [0.01, 0.05, 0.10, 1.00]   # fraction of the labelled training split
SEEDS = [0, 1, 2]
# optional overrides used only to reorder fine-tuning jobs on the pod (e.g. FT_SEEDS=1,2)
if os.environ.get("FT_BUDGETS"):
    BUDGETS = [float(v) for v in os.environ["FT_BUDGETS"].split(",")]
if os.environ.get("FT_SEEDS"):
    SEEDS = [int(v) for v in os.environ["FT_SEEDS"].split(",")]
DEDUP_HAMMING = 4         # dHash (64 bit) distance below which two images are treated as one group

# ---------------------------------------------------------------- self-supervised pretraining (DINO)
BACKBONE = "vit_small_patch16_224.dino"   # ImageNet DINO weights = starting point for domain pretraining
PT_EPOCHS = 2 if SMOKE else (30 if LOW else 100)
PT_ABL_EPOCHS = 1 if SMOKE else (6 if LOW else 30)   # leave-one-operator-out ablations
PT_BATCH = 8 if SMOKE else int(os.environ.get("PT_BATCH", 128))   # 16 GB GPUs: 96
PT_LR = 2.5e-4                            # base lr for batch 256 (scaled linearly)
PT_MIN_LR = 1e-6
PT_WD = (0.04, 0.4)
PT_WARMUP = 0 if SMOKE else (5 if LOW else 10)
PT_TEACHER_MOM = (0.996, 1.0)
PT_TEACHER_TEMP = (0.04, 0.07)            # warmed up over the first 30 epochs
PT_STUDENT_TEMP = 0.1
PT_OUT_DIM = 65536 if not SMOKE else 1024
N_LOCAL = 4 if LOW else 6
GLOBAL_SCALE = (0.32, 1.0)
LOCAL_SCALE = (0.05, 0.32)
LOCAL_SIZE = 96
P_PHYSICS = 0.8                           # prob. that the physics operator chain is applied to a view

# physics operator ranges (Table 3); severity s in [0,1] is sampled per view
PHYS = {
    "charging": {"amp": (0.0, 0.25), "sigma_frac": (0.10, 0.35)},    # low-freq additive field
    "drift": {"shift_px": (0.0, 3.0), "jitter_px": (0.0, 3.0)},        # global shift + scan-line jitter
    "psf": {"sigma": (0.3, 1.8), "aniso": (1.0, 3.0)},                 # anisotropic Gaussian (astigmatism)
    "noise": {"photons": (20.0, 400.0), "read": (0.0, 0.03)},          # Poisson-Gaussian (strong -> 20 e-/px)
    "gain": {"brightness": (-0.15, 0.15), "contrast": (0.7, 1.3), "gamma": (0.8, 1.25)},
}
PHYS_OPS = list(PHYS)

# ---------------------------------------------------------------- fine-tuning / evaluation
FT_BATCH = 8 if SMOKE else 64
FT_EPOCHS = {0.01: 2, 0.05: 2, 0.10: 2, 1.00: 2} if SMOKE else (
    {0.01: 60, 0.05: 40, 0.10: 25, 1.00: 5} if LOW else {0.01: 100, 0.05: 60, 0.10: 40, 1.00: 15})
FT_LR = {"vit": 5e-4, "cnn": 1e-3}
FT_WD = 0.05
FT_LAYER_DECAY = 0.65
FT_LABEL_SMOOTH = 0.1
FT_DROP_PATH = 0.1
FT_WARMUP_EPOCHS = 2
MIN_STEPS = 10 if SMOKE else (200 if LOW else 300)          # small budgets: run at least this many optimiser steps

# supervised "existing model" baselines (timm names, ImageNet initialisation)
BASELINES = {
    "ResNet-50": "resnet50.a1_in1k",
    "Inception-v3": "inception_v3.tv_in1k",       # architecture of Modarres et al. (2017)
    "EfficientNet-B0": "efficientnet_b0.ra_in1k",
    "ConvNeXt-T": "convnext_tiny.fb_in1k",
    "ViT-S/16 (sup. IN)": "vit_small_patch16_224.augreg_in1k",
    "DINO ViT-S/16 (IN)": "vit_small_patch16_224.dino",
}
SMOKE_MODEL = "vit_tiny_patch16_224"

KNN_K = 20
KNN_T = 0.07
LINEAR_EPOCHS = 5 if SMOKE else 100

# robustness benchmark (SEM-C): 5 artefacts x 5 fixed severities, test split only
SEMC_SEVERITIES = [1, 2, 3, 4, 5]      # s = 0.25 * severity: 4 = training max, 5 = beyond

# morphology
SYN_TRAIN = 16 if SMOKE else 400          # synthetic labelled images for the segmentation head
SYN_TEST = 8 if SMOKE else (120 if LOW else 200)
SEG_EPOCHS = 2 if SMOKE else (25 if LOW else 40)
SEG_LABEL_BUDGETS = [8, 32, 128] if not SMOKE else [4]
