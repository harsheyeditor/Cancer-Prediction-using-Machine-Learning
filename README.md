# Cancer Prediction Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5.0-orange.svg)
![Gradio](https://img.shields.io/badge/Gradio-UI-success.svg)

An industry-level, end-to-end Machine Learning project to predict breast cancer malignancy from Fine Needle Aspirate (FNA) cell measurements.

## Project Overview

This project builds a predictive model using the **Wisconsin Breast Cancer dataset**. It follows the CRISP-DM methodology and demonstrates real-world software engineering practices for ML, including modular code, reproducible pipelines, and interactive deployment.

### Key Features
- **15-Phase ML Lifecycle:** From Data Exploration to Deployment.
- **Idempotent Data Pipeline:** Modular scripts (`src/`) separate from exploratory notebooks.
- **Robust Feature Engineering:** Correlation removal and RandomForest importance selection.
- **Rigorous Evaluation:** Stratified K-Fold Cross-Validation, Precision-Recall curves, and Confusion Matrices.
- **Interactive UI:** A Gradio web app for real-time patient diagnosis.

---

## Project Structure

```text
Cancer Prediction using Machine Learning/
|-- data/
|   |-- raw/                  # Golden copy (never modified)
|   |-- processed/            # Cleaned data and feature lists
|-- images/                   # Generated plots and charts
|-- models/                   # Serialized models and scalers (.pkl)
|-- notebooks/                # Jupyter notebooks for each phase
|   |-- 01_data_exploration.ipynb
|   |-- 02_preprocessing.ipynb
|   |-- ...
|-- src/                      # Reusable Python modules
|   |-- data_loader.py
|   |-- predict.py
|   |-- preprocessing.py
|   |-- train.py
|   |-- utils.py
|-- app.py                    # Gradio Web UI
|-- requirements.txt          # Python dependencies
|-- README.md                 # Project documentation
```

---

## Getting Started

### 1. Installation & Environment Setup (Windows)

If you have downloaded this project and the `.venv` is broken or missing, follow these steps to recreate the Python environment reliably on Windows:

```powershell
# 1. Open PowerShell or Command Prompt in the project folder
cd "C:\path\to\Cancer Prediction using Machine Learning"

# 2. Delete the broken .venv folder if it exists
Remove-Item -Recurse -Force .venv

# 3. Create a fresh virtual environment using the Python Launcher
py -3.12 -m venv .venv
# (If 'py' is not recognized, use 'python -m venv .venv')

# 4. Activate the virtual environment
.\.venv\Scripts\activate

# 5. Install the required dependencies
pip install -r requirements.txt
```

### 2. Running the Data Pipeline & Training

Before running the app or notebooks, generate the required artifacts (data and models). The `train.py` script will automatically download the dataset, process it, train the model, and save the artifacts.

```powershell
python src/train.py
```

### 3. Running the Web App (Gradio)

Once the model is trained, launch the interactive diagnostic tool:

```powershell
python app.py
```
This will start a local server at `http://127.0.0.1:7860/` where you can input cell measurements and receive real-time predictions.

### 4. Exploring the Notebooks

Launch Jupyter to explore the analysis step-by-step:

```powershell
jupyter notebook
```

---

## Reproducibility and Artifacts

To maintain a clean repository and ensure reproducibility from the source, the following generated artifacts are intentionally ignored by Git:
- `data/raw/`: Raw datasets are downloaded dynamically via `sklearn` in the training pipeline.
- `data/processed/`: Cleaned data and feature lists are generated from the raw data.
- `models/`: Serialized ML models and scalers (`.pkl`) are regenerated during training.

All of these artifacts can be deterministically recreated at any time by running `python src/train.py`.

---

## Model Performance

Our final Random Forest model achieved the following performance on the test set:
- **Accuracy:** 96.5%
- **Recall (Sensitivity):** 95.2% *(Crucial for minimizing false negatives)*
- **F1 Score:** 95.2%
- **AUC-ROC:** 99.1%

---

## Contributing

Contributions are welcome! If you'd like to improve the model or add new features:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is open-source and available under the MIT License.
