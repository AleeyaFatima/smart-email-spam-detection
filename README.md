# 🛡️ SpamGuard AI: Smart Email Spam Detection System

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![UI Dashboard](https://img.shields.io/badge/frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Database](https://img.shields.io/badge/database-SQLite-003B57.svg)](https://sqlite.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade, full-stack **Smart Email Spam Detection System** featuring clean architecture, multi-model comparative machine learning analysis (TF-IDF vs Word2Vec), a FastAPI REST API backend, and a premium dark-mode custom SaaS dashboard UI.

Designed as an industry-ready portfolio project to showcase software engineering standards, clean coding, and AI/ML domain expertise.

---

## 🚀 Key Features

*   **Comparative NLP Pipeline**: Evaluates two feature extraction approaches:
    *   **TF-IDF**: Sparse statistical representation counting weighted term frequencies.
    *   **Word2Vec**: Custom-trained neural embeddings mapping semantic relationship vectors (100-dimensions).
*   **Multi-Model Analysis**: Automatically trains, tests, and evaluates **8 different combinations**:
    *   Logistic Regression, Naive Bayes (Multinomial vs. Gaussian), SVM, and Random Forest.
*   **Automated Model Selection**: Computes evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix) and dynamically designates the best-performing model (highest F1-score) as the active prediction engine.
*   **Production FastAPI Backend**: Exposes clean, validated endpoints (using Pydantic) for single predictions, bulk batch uploads, historical audit query, and retraining.
*   **Premium SaaS UI Dashboard**: Custom Dark Mode Streamlit interface featuring:
    *   Circular confidence prediction gauges and KPIs.
    *   **Keyword Trigger Inspector**: Color-highlights spam keywords dynamically.
    *   **Batch CSV Uploader**: Run spam checks on uploaded tables and download predictions.
    *   **Prediction Logs Auditor**: Full database search and filter history.
*   **Robust Database Logging**: SQLite repository pattern to save and audit lifetime queries.
*   **Zero-Setup & Offline Ready**: Auto-downloads NLTK resources and handles internet loss with fallback local training dataset generators.

---

## 📂 Folder Architecture

```
Smart Email Spam Detection System/
│
├── requirements.txt         # Package dependencies file
├── run.py                   # Central runner helper script
│
├── data/                    # Storage folders for raw and clean datasets
│   ├── raw/                 # Downloaded raw emails tsv
│   └── processed/           # Preprocessed clean_dataset.csv
│
├── models/                  # Serialized ML binaries
│   ├── metadata.json        # Evaluation metrics logs
│   ├── tfidf_vectorizer.pkl
│   └── ...                  # Model weight pickles
│
├── src/                     # Core system Python codebase
│   ├── config.py            # Settings, paths, and hyperparameters
│   ├── core/                # Clean domain entities (independent data contract)
│   ├── data_pipeline/       # Text preprocessors and splits
│   ├── ml_engine/           # Vectorizers, evaluators, trainers, and models
│   └── database/            # Connection pools and repositories
│
├── frontend/                # Streamlit Dashboard UI
│   ├── api_client.py        # Backend REST client caller
│   ├── components.py        # Premium custom HTML widgets and highlights
│   ├── styles.css           # Premium Dark Mode SaaS custom stylesheet
│   └── app.py               # Main layout page structure
│
└── docs/                    # Portfolio documentation files
    ├── folder_structure.md
    ├── installation.md
    └── project_report.md
```

---

## 🛠️ Quick Installation & Setup

Please review the [Installation Guide](file:///c:/Users/aleeya/Downloads/MY%20PROJECTS/Smart%20Email%20Spam%20Detection%20System/docs/installation.md) for full commands.

1.  **Initialize Virtual Environment**:
    ```bash
    uv venv --python 3.11
    ```
2.  **Install Requirements**:
    ```bash
    uv pip install -r requirements.txt
    ```
3.  **Train Models**:
    ```bash
    .venv\Scripts\python.exe run.py --train
    ```
4.  **Run All Services (Concurrently)**:
    ```bash
    .venv\Scripts\python.exe run.py --all
    ```

*   **Dashboard URL**: `http://127.0.0.1:8501`
*   **REST API Swagger URL**: `http://127.0.0.1:8000/docs`

---

## 📊 Core Algorithms Compared

1.  **Logistic Regression**: Weighs occurrences of words linearly to calculate a classification logit score.
2.  **Naive Bayes**: Applies Bayes theorem assuming features are conditionally independent. Uses Multinomial for TF-IDF and Gaussian for Word2Vec dense coordinates.
3.  **Support Vector Classifier (SVC)**: Finds the maximum-margin hyperplane separating spam from safe spaces.
4.  **Random Forest**: Builds an ensemble of decision trees voting on the final email class.

---

## 📃 License
Distributed under the MIT License. See `LICENSE` for more information.
