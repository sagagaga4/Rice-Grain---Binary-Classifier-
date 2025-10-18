# 🌾 Rice Grain Classifier — AI Model with TensorFlow & Keras

A **machine learning project** that classifies two types of rice grains — **Cammeo** and **Osmancik** — based on physical measurements such as *area, eccentricity, and axis length*.  
This project demonstrates **data preprocessing, normalization, visualization, and binary classification** using **TensorFlow / Keras**.

---
<img width="1399" height="727" alt="Screenshot 2025-10-18 at 21 03 36" src="https://github.com/user-attachments/assets/fc6abf93-a5ec-4380-9b99-7dd6cd7748dc" />
<img width="1415" height="752" alt="Screenshot 2025-10-18 at 21 03 27" src="https://github.com/user-attachments/assets/5fc8344d-dac4-437d-bcfb-9438a1ccd20a" />

## 🚀 Project Overview

This project builds and trains a simple **neural network classifier** to distinguish between two rice grain varieties.  
It uses real-world data from the [Google MLCC dataset](https://download.mlcc.google.com/mledu-datasets/Rice_Cammeo_Osmancik.csv).

The model learns how to differentiate rice grains based on their **shape and size characteristics**, such as:
- `Area`
- `Eccentricity`
- `Major_Axis_Length`
- and more…

---

## 📊 Features

✅ Data loading and preprocessing from a live dataset  
✅ Data exploration with **Plotly** (2D and 3D scatter plots)  
✅ Feature normalization (Z-score scaling)  
✅ Training/validation/test split (80/10/10)  
✅ **Keras Sequential model** with `ReLU` and `Sigmoid` activations  
✅ Model performance visualization: Accuracy, Precision, Recall, AUC  
✅ Evaluation on unseen test data  

---

## 🧠 Model Architecture

The model is a simple feed-forward neural network:

| Layer | Type | Activation | Units |
|-------|------|-------------|-------|
| Input | Dense | — | 3 |
| Hidden 1 | Dense | ReLU | 8 |
| Hidden 2 | Dense | ReLU | 4 |
| Output | Dense | Sigmoid | 1 |

**Loss:** Binary Crossentropy  
**Optimizer:** RMSprop  
**Metrics:** Accuracy, Precision, Recall, AUC  

---

## 🧩 Technologies Used

- 🐍 **Python 3**
- 🧠 **TensorFlow / Keras**
- 📈 **Plotly Express** & **Plotly Graph Objects**
- 🧮 **NumPy & Pandas**
- 💻 **Jupyter / VSCode / Command-line**

---

## ⚙️ Installation & Running

Clone the project and install dependencies:

```bash
git clone https://github.com/yourusername/rice-grain-classifier.git
cd rice-grain-classifier
pip install tensorflow pandas numpy plotly
python3 model.py
