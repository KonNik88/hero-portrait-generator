# Hero Portrait Generator — Conditional GAN (R&D Project)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]()
[![GAN](https://img.shields.io/badge/Model-cGAN-blueviolet.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

A research-oriented deep-learning study exploring **conditional generation of pixel-art fantasy hero portraits** under the constraints of:
- a *small dataset* (~7,600 images),
- *high inter-class variation*,
- and visually demanding pixel-art style.

The project demonstrates what modern **conditional GANs** can and cannot achieve in such conditions.

---

## Goal

To generate hero portraits conditioned on:

- gender (`masculine`, `feminine`)
- hero class (`warrior`, `mage`, `rogue`, `cavalry`, …)
- palette category (`warm`, `cool`, `dark`, `neutral`, `muted`)

Each image was preprocessed and labeled automatically + manually curated where needed.

---

## Implemented Models & Techniques

### **Architectures**
- Conditional DCGAN (baseline)
- ResNet-based G & D
- Conditional embeddings for all attributes
- Projection Discriminator (class conditioning)

### **Stabilization**
- Spectral Normalization  
- Gradient Penalty (R1)  
- EMA on Generator  
- TTUR (different LR for G and D)  
- DiffAugment  
- Instance Noise  

### **Resolution Experiments**
- Training at **64×64**
- Palette conditioning ON/OFF

---

## Findings

### ✔ GAN learns stable silhouettes  
Across runs, the generator consistently captures head/shoulder shapes and rough armor/clothing structure.

### ✔ Palette categories influence global tone  
Without palette conditioning, global hues drift more freely; with palette labels, tones become more consistent.

### ❗ Fine-detail failure  
Due to the small dataset and strong style diversity:
- details (eyes, mouth, helmets) blur,
- local textures collapse,
- artifacts appear frequently.

---

## Final Conclusion

This project clearly demonstrates the **practical limits of conditional GANs on small, heterogeneous pixel-art datasets**:

- Stable training is possible.
- Coarse structure is learnable.
- High-detail, style-faithful generation is *not* achievable without:
  - significantly more data, or  
  - much heavier architectures (StyleGAN2, DiT, DDPM) that exceed RTX 2070 compute.

**Therefore:**  
The project stands as a **research case study**, not a production-ready generator — and that is completely valid for portfolio and education.

---

## Repository Structure
hero-portrait-generator/
  data/                        # prepared data
  train_images/                # not tracked
  notebooks/
    01_prepare_dataset.ipynb
    02_build_final_dataset.ipynb
    03_train_cdcgan.ipynb
  artifacts_gan/
    samples/
    checkpoints/
  README.md
  requirements.txt
  .gitignore
  LICENSE

---

## Usage
pip install -r requirements.txt

Outputs appear in `artifacts_gan/samples` and `artifacts_gan/checkpoints`.

---

## Requirements

torch>=2.0
torchvision>=0.15
numpy
pandas
Pillow
matplotlib
tqdm

---

## License

MIT License.
