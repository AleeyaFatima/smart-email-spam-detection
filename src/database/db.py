import sqlite3
import logging
from src.config import DB_PATH

logger = logging.getLogger(__name__)

# ==========================================
# DESIGN DECISION: SQLite for Portfolio Projects
# We use SQLite for the database because:
# 1. It requires zero configuration, zero servers to install, and zero setup.
# 2. It is highly standard and stores all records in a single local file (`database.db`).
# 3. It is perfect for demonstrating CRUD repository design patterns.
# ==========================================

def get_db_connection():
    """
    Creates and returns a connection to the SQLite database.
    Configures row factory to return records as dictionaries instead of tuples,
    making database outputs extremely easy to map to Pydantic schemas.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Creates the database tables if they do not exist.
    """
    logger.info(f"Initializing SQLite database at: {DB_PATH}")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create the prediction_history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email_text TEXT NOT NULL,
            is_spam INTEGER NOT NULL,
            probability REAL NOT NULL,
            confidence REAL NOT NULL,
            model_used TEXT NOT NULL,
            vectorizer_used TEXT NOT NULL,
            execution_time_ms REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully.")

if __name__ == "__main__":
    init_db()
