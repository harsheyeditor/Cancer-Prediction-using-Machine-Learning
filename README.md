# 🧬 Cancer Prediction Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5.0-orange.svg)
![Gradio](https://img.shields.io/badge/Gradio-UI-success.svg)

An industry-level, end-to-end Machine Learning project to predict breast cancer malignancy from Fine Needle Aspirate (FNA) cell measurements.

## 🎯 Project Overview

This project builds a predictive model using the **Wisconsin Breast Cancer dataset**. It follows the CRISP-DM methodology and demonstrates real-world software engineering practices for ML, including modular code, reproducible pipelines, and interactive deployment.

### 🌟 Key Features
- **15-Phase ML Lifecycle:** From Data Exploration to Deployment.
- **Idempotent Data Pipeline:** Modular scripts (`src/`) separate from exploratory notebooks.
- **Robust Feature Engineering:** Correlation removal and RandomForest importance selection.
- **Rigorous Evaluation:** Stratified K-Fold Cross-Validation, Precision-Recall curves, and Confusion Matrices.
- **Interactive UI:** A Gradio web app for real-time patient diagnosis.

---

## 📁 Project Structure

```text
Cancer Prediction using Machine Learning/
├── data/
│   ├── raw/                  # Golden copy (never modified)
│   └── processed/            # Cleaned data and feature lists
├── images/                   # Generated plots and charts
├── models/                   # Serialized models and scalers (.pkl)
├── notebooks/                # Jupyter notebooks for each phase
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   ├── 06_evaluation.ipynb
│   └── 07_prediction.ipynb
├── src/                      # Reusable Python modules
│   ├── data_loader.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── train.py
│   └── utils.py
├── app.py                    # Gradio Web UI
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and install the dependencies:

```bash
# Clone repo
git clone https://github.com/yourusername/cancer-prediction-ml.git
cd cancer-prediction-ml

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Running the Notebooks

Launch Jupyter to explore the analysis step-by-step:

```bash
jupyter notebook
```

*Note: Ensure you select the kernel `Python 3 (Cancer Prediction)` if prompted.*

### 3. Running the Web App (Gradio)

To launch the interactive diagnostic tool:

```bash
python app.py
```
This will start a local server at `http://127.0.0.1:7860/` where you can input cell measurements and receive real-time predictions.

### 4. Training the Model from CLI

To retrain the model and regenerate the artifacts in the `models/` directory without using notebooks:

```bash
python src/train.py
```

---

## 📊 Model Performance

Our final Random Forest model achieved the following performance on the test set:
- **Accuracy:** 96.5%
- **Recall (Sensitivity):** 95.2% *(Crucial for minimizing false negatives)*
- **F1 Score:** 95.2%
- **AUC-ROC:** 99.1%

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the model or add new features:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open-source and available under the MIT License.
