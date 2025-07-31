# XLFM-Former: Physics-Consistent 3D Reconstruction for Light Field Microscopy

This repository provides a minimal and reproducible implementation of **XLFM-Former**, a transformer-based framework for 3D reconstruction from light field microscopy (XLFM) data. It is designed to address reproducibility and clarity concerns raised in the NeurIPS 2025 reviews.

## 🔍 Overview

XLFM-Former combines:

- **Masked View Modeling for Light Fields (MVM-LF)**: a self-supervised pretraining strategy that learns angular dependencies by reconstructing masked views;
- **Optical Rendering Consistency (ORC) Loss**: a physics-grounded loss that enforces agreement between reconstructed volumes and their PSF-based forward projections.

## 🚀 Quickstart

We provide a toy setup for reproducibility. You can simulate light fields and test ORC loss behavior:

```bash
# Step 1: Install dependencies
pip install -r requirements.txt
```



### 🔬 Inference 

Run the inference script with the following command:

```bash
bash
python ./inference.py --config ./configs/infer/Nemos/xx.yaml
```

### 🏋️‍♂️ Training 

To start the training process, use the following command:

```bash
bash
python ./main.py --config ./configs/NemoS/main_mid/Others/xx.yaml
```

The results will be saved to the output directory as specified in the configuration file.





📦 Dependencies

- `torch>=1.10`
- `einops`
- `numpy`
- `matplotlib`

See `requirements.txt` for full list.



## 🔒 Notes on Anonymity

This repository is anonymized for review and does not contain dataset download links or institution-specific metadata. Full dataset and codebase will be released upon publication.
