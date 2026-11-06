<div align="center">

# CXR-Trust

### Calibrated and Explainable Multi-Label Chest X-ray Classification

[Dang Truong Minh Le](https://github.com/dangaura-tech)

Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)

[![CI](https://github.com/dangaura-tech/cxr-trust/actions/workflows/ci.yml/badge.svg)](https://github.com/dangaura-tech/cxr-trust/actions/workflows/ci.yml)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

---

## Abstract

Deep networks for chest radiograph interpretation report high AUROC, but two properties
matter just as much for clinical use: **calibration** (does a predicted 0.8 mean an 80 %
chance?) and **localisation** (is the model looking at the lung, or at a hospital token?).
**CXR-Trust** is a research framework that trains multi-label classifiers on CheXpert and
NIH ChestX-ray14 and evaluates them on all three axes - discrimination, calibration and
explanation - in one reproducible pipeline:

* **Label uncertainty** is handled explicitly with the U-Ones / U-Zeros / U-Ignore policies,
  where U-Ignore masks uncertain labels out of a *masked* BCE or asymmetric loss.
* **Patient-level splits** prevent the leakage that inflates results when images of the
  same patient appear in both train and validation sets.
* **Per-class temperature scaling** is fitted on the validation split and reported with
  ECE, Brier score and reliability curves. Because it does not change the ranking, AUROC
  stays the same.
* **Grad-CAM / Grad-CAM++** maps are produced for the top-k predicted findings of any image,
  and their **faithfulness** (deletion / insertion) and **localisation** (pointing game,
  box energy on the NIH expert boxes) are measured against a random-saliency control.
* **Distribution-free uncertainty.** Label-conditional (Mondrian) conformal prediction gives
  every finding a set over {absent, present} whose coverage holds *for each label*, so rare
  findings are protected too. Conformal risk control picks a reporting threshold with a
  guaranteed bound on the per-image **false-negative rate**. Risk-coverage curves (AURC,
  E-AURC) quantify how well uncertainty supports **triage by abstention**.
* **Zero-shot vision-language baseline.** A biomedical CLIP model (BiomedCLIP by default)
  scores each finding by contrasting ensembles of positive and negative prompts. Its logits go
  through exactly the same calibration and conformal protocol as the trained CNNs, so
  supervised and zero-shot models are compared on equal terms.

## Framework

```
 CXR ─► Backbone (DenseNet-121 | ResNet-50 | EfficientNet-B0 | ConvNeXt-T) ─► GAP ─► Linear(14)
         │  (EMA weights, AMP, AdamW + warmup-cosine, discriminative LR for the head)
         │
         ├─► Masked BCE / Asymmetric loss   ◄── CheXpert labels + uncertainty policy
         ├─► per-class temperature  T_k      ◄── fitted on validation half A (L-BFGS, NLL)
         ├─► conformal sets / FNR control    ◄── calibrated on validation half B
         ├─► risk-coverage (abstention)      ◄── entropy or min-margin confidence
         └─► Grad-CAM(++) on last conv block ◄── deletion / insertion, pointing game

 CXR ─► BiomedCLIP image encoder ─┐
        "a chest x-ray showing {finding}" / "no evidence of {finding}" ─► text encoder
                                  └─► s·(cos⁺ − cos⁻) = zero-shot logit ─► same protocol as above
```

The validation split is divided into two disjoint halves: one fits the temperatures, the
other calibrates the conformal procedures on the temperature-scaled probabilities. This keeps
the finite-sample guarantees valid.

## Installation

```bash
git clone https://github.com/dangaura-tech/cxr-trust.git
cd cxr-trust
pip install -r requirements.txt   # or: pip install -e ".[dev]"
```

## Data

No images or labels are included. Request access and download the data from the
official sources, then point the configs at them:

| Dataset | Source | Expected path |
|---|---|---|
| CheXpert-v1.0-small | [Stanford ML Group](https://stanfordmlgroup.github.io/competitions/chexpert/) | `data/CheXpert-v1.0-small/{train,valid}.csv` |
| NIH ChestX-ray14 | [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC) | `data/nih/images/`, `data/nih/Data_Entry_2017.csv` |
| NIH bounding boxes (optional) | same release | `data/nih/BBox_List_2017.csv` |

`bash scripts/check_data.sh` verifies the layout.

## Usage

```bash
# 1) train (checkpoint with EMA weights -> runs/<config>/best.pth)
python tools/train.py --config configs/chexpert_densenet121.yaml

# 2) evaluate: AUROC / AUPRC, ECE / Brier before and after temperature scaling,
#    conformal coverage per finding, FNR control and risk-coverage at risk level alpha
python tools/evaluate.py --checkpoint runs/chexpert_densenet121/best.pth --alpha 0.1 \
    --out results/chexpert_dn121.json --save-logits results/chexpert_dn121_logits.npz

# 3) explain a single radiograph
python tools/explain.py --checkpoint runs/chexpert_densenet121/best.pth \
    --image path/to/cxr.jpg --topk 3 --plus-plus --output explanations/

# 4) faithfulness and localisation of the explanations (NIH boxes are optional)
python tools/faithfulness.py --checkpoint runs/nih_densenet121/best.pth --limit 200 \
    --bbox-csv data/nih/BBox_List_2017.csv --out results/faithfulness.json

# 5) zero-shot vision-language baseline, no training (downloads BiomedCLIP on first use)
pip install -e ".[vlm]"
python tools/zeroshot.py --config configs/chexpert_densenet121.yaml --out results/zeroshot.json
```

Any config value can be overridden on the command line, e.g.
`python tools/train.py --config configs/nih_densenet121.yaml model.backbone=convnext_tiny train.lr=5e-5`.

### Provided experiment configs

| Config | Backbone | Loss | Uncertainty policy |
|---|---|---|---|
| `chexpert_densenet121.yaml` | DenseNet-121 | BCE | U-Ones |
| `chexpert_uignore.yaml` | DenseNet-121 | masked BCE | U-Ignore |
| `chexpert_asl_convnext.yaml` | ConvNeXt-T | Asymmetric loss | U-Ones |
| `nih_densenet121.yaml` | DenseNet-121 | class-balanced BCE | - |

## Repository structure

```
cxr-trust/
├── cxrtrust/
│   ├── data/            # CheXpert / NIH loaders, uncertainty policies, patient split, transforms
│   ├── models/          # torchvision backbones with a multi-label head
│   ├── losses.py        # masked BCE, asymmetric loss
│   ├── calibration.py   # temperature scaling, ECE, Brier, reliability curve
│   ├── conformal.py     # Mondrian conformal sets, conformal FNR control
│   ├── selective.py     # confidence scores, risk-coverage, AURC / E-AURC
│   ├── gradcam.py       # Grad-CAM and Grad-CAM++
│   ├── faithfulness.py  # deletion / insertion, pointing game, NIH boxes
│   ├── zeroshot.py      # prompt-ensembled zero-shot VLM classifier (BiomedCLIP / open_clip)
│   ├── report.py        # shared evaluation protocol for any logits
│   ├── metrics.py       # per-class AUROC / AUPRC, F1-optimal thresholds
│   └── engine.py        # trainer with EMA + AMP
├── configs/
├── tools/               # train / evaluate / explain / faithfulness / zeroshot
├── scripts/
└── tests/
```

## Citation

```bibtex
@misc{le2026cxrtrust,
  title        = {{CXR-Trust}: Calibrated and Explainable Multi-Label Chest X-ray Classification},
  author       = {Le, Dang Truong Minh},
  year         = {2026},
  howpublished = {\url{https://github.com/dangaura-tech/cxr-trust}}
}
```

## References

* Irvin et al., *CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison*, AAAI 2019.
* Wang et al., *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks*, CVPR 2017.
* Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017.
* Selvaraju et al., *Grad-CAM*, ICCV 2017; Chattopadhay et al., *Grad-CAM++*, WACV 2018.
* Ridnik et al., *Asymmetric Loss for Multi-Label Classification*, ICCV 2021.
* Vovk et al., *Algorithmic Learning in a Random World*, 2005 · Angelopoulos et al., *Conformal Risk Control*, ICLR 2024.
* Geifman & El-Yaniv, *Selective Classification for Deep Neural Networks*, NeurIPS 2017.
* Petsiuk et al., *RISE: Randomized Input Sampling for Explanation*, BMVC 2018 · Zhang et al., *Top-down Neural Attention by Excitation Backprop*, IJCV 2018.
* Radford et al., *CLIP*, ICML 2021 · Tiu et al., *CheXzero*, Nature BME 2022 · Zhang et al., *BiomedCLIP*, 2023.

## Disclaimer

This is research software. It is **not** a medical device and has not been validated for
clinical use.

## License

MIT
