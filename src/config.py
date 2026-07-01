import os
from pathlib import Path

# ==========================================
# DESIGN DECISION: Centralized Configuration
# We keep all directories, database paths, and model parameters in one place
# so that the backend, frontend, and ML pipeline always agree.
# Using Pathlib makes paths work on both Windows and Linux without changes.
# ==========================================

# Base Directories
# Path(__file__).resolve() finds the absolute path of this file.
# .parents[1] goes up two levels (from src/config.py -> src -> Smart Email Spam Detection System)
BASE_DIR = Path(__file__).resolve().parents[1]

# Data Directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model Directories (where we save the trained pickle files)
MODEL_DIR = BASE_DIR / "models"

# Database path (SQLite file)
if os.environ.get("VERCEL"):
    DB_PATH = Path("/tmp") / "database.db"
else:
    DB_PATH = BASE_DIR / "database.db"

# Create directories if they do not exist
for folder in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR]:
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

# Dataset URLs and Fallbacks
# We use a reliable SMS/Email spam dataset hosted on raw github for download.
DATASET_URL = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
RAW_DATA_FILENAME = "spam_dataset.csv"
RAW_DATA_PATH = RAW_DATA_DIR / RAW_DATA_FILENAME
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "clean_dataset.csv"

# Model Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2  # 20% of data is used for testing, 80% for training

# Web Server Settings
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"

# NLTK Data path (where stopword lists are downloaded)
# Storing NLTK data inside the project directory makes the app self-contained and portable.
NLTK_DATA_DIR = BASE_DIR / "nltk_data"
try:
    NLTK_DATA_DIR.mkdir(exist_ok=True)
except OSError:
    pass
os.environ["NLTK_DATA"] = str(NLTK_DATA_DIR)

# Logging Format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = BASE_DIR / "app.log"
