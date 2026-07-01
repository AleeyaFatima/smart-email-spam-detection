# Project Directory Layout & File Explanations
## Smart Email Spam Detection System

This document explains the organization of files inside this repository and the purpose of every single component.

---

### Folder Tree
```
Smart Email Spam Detection System/
│
├── .gitignore               # Ignores virtual env, database, and cache files in Git
├── requirements.txt         # Package dependencies file (FastAPI, Streamlit, etc.)
├── run.py                   # Central runner helper script
│
├── data/                    # Storage folders for raw and clean datasets
│   ├── raw/
│   │   └── spam_dataset.csv
│   └── processed/
│       └── clean_dataset.csv
│
├── models/                  # Storage folder for trained machine learning files
│   ├── metadata.json        # Tracks model comparison metrics and active model
│   ├── tfidf_vectorizer.pkl
│   ├── word2vec_model.bin
│   └── ...                  # Serialized .pkl binaries for classifiers
│
├── src/                     # Core system Python codebase
│   ├── __init__.py
│   ├── config.py            # Central settings, paths, and hyperparameters
│   │
│   ├── core/                # Clean architecture domain models
│   │   ├── __init__.py
│   │   └── entities.py      # Dataclasses (EmailMessage, PredictionResult, etc.)
│   │
│   ├── data_pipeline/       # Data preprocessing and splits
│   │   ├── __init__.py
│   │   ├── preprocessor.py  # NLTK text scrub, stopwords filter, lemmatization
│   │   └── loader.py        # Dataset downloader and train/test splits
│   │
│   ├── ml_engine/           # Algorithms training and evaluations
│   │   ├── __init__.py
│   │   ├── vectorizers.py   # Wrapped TF-IDF and Word2Vec class vectors
│   │   ├── evaluator.py     # Calculates accuracy, precision, recall, F1
│   │   ├── models.py        # SpamClassifier prediction pipeline wrapper
│   │   └── trainer.py       # Rebuilds and compares all 8 configurations
│   │
│   └── database/            # Local SQLite database layer
│       ├── __init__.py
│       ├── db.py            # Connection pool and table creators
│       └── repository.py    # SQL query interfaces (insert, stats, search history)
│
├── frontend/                # Streamlit Dashboard files
│   ├── api_client.py        # Communicates with FastAPI backend REST routes
│   ├── components.py        # Renders KPI cards, gauges, and highlighted text
│   ├── styles.css           # Premium Dark Mode SaaS custom stylesheet
│   └── app.py               # Main layout page structure and Plotly visual maps
│
└── docs/                    # Portfolio documentation files
    ├── folder_structure.md
    ├── installation.md
    └── project_report.md
```

---

### Detailed File Summary

#### 1. Core Domain Layer (`src/core/`)
*   **`entities.py`**: Declares native Python data templates (`EmailMessage`, `PredictionResult`, `ModelMetrics`) that act as communication contracts between the database, ML engine, and API routes.

#### 2. Data Pipeline Layer (`src/data_pipeline/`)
*   **`preprocessor.py`**: Performs natural language text cleaning: lowercases letters, strips HTML/URLs, scrubs numbers/punctuation, tokenizes sentences, filters out NLTK stopwords, and lemmatizes remaining words to their roots.
*   **`loader.py`**: Handles dataset inputs. Downloads raw CSV or falls back to standard local samples if offline. Preprocesses texts, generates `clean_dataset.csv`, and performs stratified splits.

#### 3. Machine Learning Layer (`src/ml_engine/`)
*   **`vectorizers.py`**: Translates text into float matrices using TF-IDF weights or averaged custom Word2Vec embeddings.
*   **`evaluator.py`**: Scores guesses using Scikit-Learn evaluation functions.
*   **`models.py`**: Encapsulates model loads and prediction execution, calculating latency and confidence scores.
*   **`trainer.py`**: Fits all 8 vectorizer-algorithm permutations, saves files to `/models`, and designates the best configuration.

#### 4. Storage & API Layer (`src/database/` & `src/backend/`)
*   **`db.py`**: Instantiates local SQLite database connection details.
*   **`repository.py`**: Performs SQL inserts, log audit lookups, and aggregate KPI sum math.
*   **`schemas.py`**: Handles Pydantic type safety validations.
*   **`routes.py`**: Directs web routes for predictions, batch scans, log auditing, and retraining.
*   **`app.py`**: Initializes FastAPI, hooks CORS permissions, and executes database table creation on startup.

#### 5. User Interface Layer (`frontend/`)
*   **`api_client.py`**: Makes requests to the backend APIs.
*   **`components.py`**: Generates high-fidelity HTML cards, circular prediction meters, and word analyzer highlight wrappers.
*   **`styles.css`**: Styling sheets for Outfit font displays, card animations, and button glows.
*   **`app.py`**: The multi-tab Streamlit dashboard controller.
