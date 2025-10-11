# Hero Portrait Generator — GAN/VAE/DDPM

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-brightgreen.svg)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-teal.svg)](#)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

A creative deep-learning project for generating **fantasy-style hero portraits** (retro pixel-art / RPG vibe) under modest compute (e.g., RTX 2070).  
The repo showcases a **complete generative pipeline**: data prep → models (VAE, DCGAN, WGAN-GP, VQ-VAE, lightweight DDPM) → **conditional** sampling with user-controlled attributes → evaluation (FID/KID, conditional accuracy) → **deployment** (FastAPI + Streamlit) → Docker.

> **Why**: more exciting than MNIST/LFW, portfolio-ready, and actually fun to use.

---

## Features
- **Data preparation**: image normalization (64×64), palette quantization (optional), attribute tags (class, pose, gear, palette).
- **Models**:
  - Baselines: **VAE**, **DCGAN**, **WGAN-GP** (+ EMA)
  - Discrete codes: **VQ-VAE** (+ optional PixelCNN prior)
  - Diffusion (compact): **DDPM-lite (64×64 UNet)** with classifier-free guidance for attributes
- **Conditional generation**:
  - Attribute controls: `class ∈ {warrior, mage, rogue, ...}`, `pose ∈ {frontal, 3/4}`, `helmet`, `hood`, `beard`, `palette ∈ {warm, cold, mono}`, etc.
  - Mini-DSL / validated prompt: `class=warrior; helmet; beard; palette=warm; pose=frontal`
- **Evaluation**: FID/KID; attribute consistency via a small attribute-classifier
- **Apps**:
  - **Streamlit UI**: sliders/toggles/presets, batch grids, latent walks, PNG/ZIP download
  - **FastAPI**: `/generate`, `/interpolate` JSON endpoints
- **Ops**: MLflow logging, artifacts folder, Dockerfiles for API/UI, Compose

---

##  Project Structure (planned)
```
hero-portrait-generator/
  data/
    raw/                # raw images (not tracked)
    processed/          # 64x64 aligned images + tags
    scripts/            # kaggle/processing scripts
  models/
    vae.py
    dcgan.py
    wgan_gp.py
    vqvae.py
    ddpm_unet.py
  train/
    train_vae.py
    train_dcgan.py
    train_wgan_gp.py
    train_vqvae.py
    train_ddpm.py
  inference/
    sample.py           # generate grids / latent walks
    cond_utils.py
  metrics/
    fid.py
    kid.py
    attr_eval.py
  api/
    main.py             # FastAPI
    schemas.py
  ui/
    streamlit_app.py
  docker/
    Dockerfile.api
    Dockerfile.ui
    docker-compose.yml
  configs/
    default.yaml
  artifacts/            # saved checkpoints, mlflow runs
  README.md
  requirements.txt
  LICENSE
```

---

## Quickstart

### 0) Environment
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

### 1) Data
- Put hero portraits under `data/raw/` (or use provided download/convert scripts).
- Run preprocessing (resize → align → palette quantization (optional) → tag CSV):
```bash
python data/scripts/prepare_data.py   --in_dir data/raw --out_dir data/processed --size 64 --make_tags
```

### 2) Train (example: WGAN-GP cond, 64×64)
```bash
python train/train_wgan_gp.py   --data_dir data/processed   --attrs class pose helmet hood beard palette   --batch_size 128 --epochs 100 --lr 2e-4 --mlflow
```

### 3) Evaluate (FID, KID, attribute consistency)
```bash
python metrics/fid.py --real data/processed --fake artifacts/samples
python metrics/kid.py --real data/processed --fake artifacts/samples
python metrics/attr_eval.py --fake artifacts/samples --attr_clf artifacts/attr_clf.pt
```

### 4) Inference
```bash
python inference/sample.py   --checkpoint artifacts/wgan_gp_gen.pt   --class warrior --pose frontal --helmet --beard --palette warm   --seed 123 --truncation 0.6 --n 16 --out artifacts/samples/grid.png
```

---

## Streamlit UI
```bash
streamlit run ui/streamlit_app.py
```
**Controls**: class/pose (select), helmet/hood/beard (toggles), palette (select), diversity/truncation & seed (sliders), **Generate**, **x16 Grid**, **Latent Walk**, **Download**.

---

## FastAPI
```bash
uvicorn api.main:app --reload
```
**POST** `/generate`
```json
{
  "class": "warrior",
  "pose": "frontal",
  "helmet": true,
  "beard": true,
  "palette": "warm",
  "seed": 123,
  "truncation": 0.6,
  "n": 8
}
```
Returns base64 PNG(s) or file paths.

---

## Models Roadmap
- [x] VAE (sanity check)
- [x] DCGAN / [x] WGAN-GP (+ EMA, projection D)
- [ ] VQ-VAE (+ PixelCNN prior)
- [ ] DDPM-lite (64×64 UNet, CFG for attributes)
- [ ] Attribute-classifier for conditional accuracy
- [ ] Palette-transfer post-processing

---

## Logging & Reproducibility
- **MLflow** runs with params/metrics/artifacts (`mlruns/` ignored by git).
- Global seeds via `torch.manual_seed` + `numpy.random.seed`.
- Config-first training (`configs/default.yaml`).
- Deterministic cuDNN flags (when feasible).

---

## Docker (sketch)
```bash
# API
docker build -f docker/Dockerfile.api -t hero-api .
# UI
docker build -f docker/Dockerfile.ui  -t hero-ui  .
# Compose (API + UI)
docker compose -f docker/docker-compose.yml up
```

---

## License
This project is released under the **MIT License**. See [LICENSE](./LICENSE).

> **Note** on datasets: ensure you have the right to use and redistribute the images you train on.  
> If using public datasets, respect their licenses and trademark policies.

---

## Suggested GitHub Topics
`deep-learning`, `gan`, `vae`, `ddpm`, `generative-models`, `computer-vision`, `pixel-art`, `rpg`, `game-dev`, `streamlit`, `fastapi`, `mlops`, `portfolio-project`

---

## Acknowledgements
Thanks to the open-source community for reusable building blocks (PyTorch, TorchMetrics, Streamlit, FastAPI).  
Palette quantization ideas inspired by classic pixel-art workflows.
