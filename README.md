# 🩸 HEMA-AI — Peripheral Blood Cell Classifier

A Streamlit web app that classifies microscopic peripheral blood cell images into 8 cell types using a VGG16 transfer-learning model.

**Live app:** https://bloodcellpredictionapp.streamlit.app/

## Overview

Upload a microscopic image of a peripheral blood cell and the app returns:
- The predicted cell type with a confidence score
- A full probability breakdown across all 8 classes
- A short clinical description and typical reference range for the predicted cell
- A reference "Cell Atlas" tab describing all 8 classes
- A "System Details" tab describing the model architecture

The 8 recognized classes are:

| Class | Category |
|---|---|
| Basophil | Granulocyte |
| Eosinophil | Granulocyte |
| Erythroblast | Red cell precursor |
| Immature Granulocyte (IG) | Granulocyte precursor |
| Lymphocyte | Agranulocyte |
| Monocyte | Agranulocyte |
| Neutrophil | Granulocyte |
| Platelet (Thrombocyte) | Cell fragment |

> ⚠️ **Disclaimer:** This app is for research and educational demonstration only. It is **not** a diagnostic medical device and should not be used for clinical decision-making.

## Model

- **Architecture:** VGG16 (ImageNet-pretrained convolutional base) + a custom classification head, trained via transfer learning
- **Input:** 224 × 224 × 3 RGB images
- **Preprocessing:** Keras VGG16 `preprocess_input` normalization, plus light data augmentation (flip, rotation, zoom) during training
- **Output:** Softmax probabilities over the 8 cell classes
- **Dataset:** PBC dataset of normal peripheral blood cells, split 70% train / 15% validation / 15% test
- **Performance:** ~88.5% test accuracy (vs. ~36% for a custom CNN baseline trained on the same split — see `PBC.ipynb` for the full comparison, training curves, and classification report)

Model training and evaluation are documented in [`PBC.ipynb`](PBC.ipynb). The trained weights used by the app are stored in `artifacts/vgg16_pcb.keras`, with class labels in `artifacts/class_names.json`.

## Project Structure

```
Blood_Cell_Prediction_App/
├── app.py                     # Streamlit application
├── PBC.ipynb                  # Model training & evaluation notebook
├── requirements.txt           # Python dependencies
├── runtime.txt                # Python version pin (for deployment)
└── artifacts/
    ├── vgg16_pcb.keras        # Trained VGG16 model weights
    └── class_names.json       # Ordered list of class labels
```

## Getting Started

### Prerequisites

- Python 3.11 (see `runtime.txt`)

### Installation

```bash
git clone https://github.com/arya-arun-123/Blood_Cell_Prediction_App.git
cd Blood_Cell_Prediction_App
pip install -r requirements.txt
```

### Run locally

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`) in your browser.

### Usage

1. Go to the **Image Classification** tab.
2. Upload a JPEG/PNG image of a single peripheral blood cell (microscopic magnification).
3. Click **Run Cellular Analysis**.
4. View the predicted cell type, confidence score, and full probability distribution.

## Tech Stack

- [Streamlit](https://streamlit.io/) — web app framework
- [TensorFlow / Keras](https://www.tensorflow.org/) — model training & inference (VGG16 transfer learning)
- [Pillow](https://python-pillow.org/) — image handling
- [NumPy](https://numpy.org/) — array/numeric operations

