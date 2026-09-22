# DHWANI - Wake Word Detection Model 🛰️🎙️

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FVAh6uvgVPN77TXtuqtKs_jwRiq7dZwG?usp=sharing)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)

**DHWANI** is a lightweight, edge-friendly **Depthwise Separable Convolutional Neural Network (DS-CNN)** designed for real-time wake word detection (`"HEY ISRO"` vs `"NOT HEY ISRO"`).

---

## 📌 Features

- **Lightweight Architecture**: Depthwise Separable Convolutions (DS-CNN) ensure high computational efficiency and minimal parameter footprint, ideal for embedded and edge hardware.
- **Audio Preprocessing**: Standardized to 16 kHz mono audio, converted into 40x40 Log-Mel Spectrograms.
- **Regularization & Stability**: Feature augmentation via Gaussian noise, Batch Normalization, and Dropout to mitigate overfitting on speech samples.
- **High Accuracy**: Reaches over 98% accuracy on target speech datasets.

---

## 🏗️ Model Architecture

The model processes **40 × 40 × 1** Log-Mel feature maps:

```text
Input (40, 40, 1)
      │
 Gaussian Noise (0.03)
      │
 Conv2D (16 filters, 3x3, stride=2, padding="same") + BatchNorm + ReLU
      │
 DSBlock (16 filters, stride=1)
      │
 DSBlock (24 filters, stride=2)
      │
 DSBlock (32 filters, stride=2)
      │
 DSBlock (48 filters, stride=2)
      │
 GlobalAveragePooling2D
      │
 Dropout (0.35)
      │
 Dense (2 units, activation="softmax")
```

Each **DSBlock** contains:
1. `DepthwiseConv2D(3x3, stride, padding="same")` + `BatchNormalization` + `ReLU`
2. `Conv2D(filters, 1x1, padding="same")` + `BatchNormalization` + `ReLU`

---

## 📁 Repository Structure

```text
DHWANI-MODEL/
├── dhwani_model.ipynb     # Original Google Colab notebook with outputs & training logs
├── train.py               # Standalone Python script for training and evaluation
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore rules for dataset, weights, and caches
└── README.md              # Project documentation
```

---

## 🚀 Getting Started

### 1. Open directly in Google Colab
You can run the notebook directly in your browser:
👉 [Open in Google Colab](https://colab.research.google.com/drive/1FVAh6uvgVPN77TXtuqtKs_jwRiq7dZwG?usp=sharing)

### 2. Local Setup & Training

#### Clone the repository:
```bash
git clone https://github.com/Vijayaramanujam/DHWANI-MODEL.git
cd DHWANI-MODEL
```

#### Install dependencies:
```bash
pip install -r requirements.txt
```

#### Prepare your dataset:
Organize your audio files (`.wav`, 16 kHz mono recommended) in the following structure:
```text
ISRO_WakeWord_WAV/
├── 01_HEY_ISRO/
│   ├── sample1.wav
│   └── ...
└── 02_NOT_HEY_ISRO/
    ├── sample1.wav
    └── ...
```

#### Run Training:
```bash
python train.py --dataset_path ISRO_WakeWord_WAV --epochs 30 --batch_size 16 --lr 0.0005
```

---

## 📊 Preprocessing Parameters

| Parameter | Value |
| :--- | :--- |
| **Sampling Rate (SR)** | 16,000 Hz (16 kHz) |
| **Audio Duration** | 1.0 second (padded / trimmed) |
| **Mel Filterbanks (`n_mels`)** | 40 |
| **FFT Size (`n_fft`)** | 512 |
| **Hop Length** | 400 |
| **Spectrogram Dimension** | 40 × 40 |

---

## 📄 License

This repository is maintained for the DHWANI Wake Word Detection Project.
