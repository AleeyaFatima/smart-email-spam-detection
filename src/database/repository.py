import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from src.database.db import get_db_connection
from src.core.entities import PredictionResult

logger = logging.getLogger(__name__)

# ==========================================
# DESIGN DECISION: Repository Pattern
# By encapsulating SQL queries inside a 'PredictionRepository' class,
# the rest of the application (backend, routing) doesn't write SQL queries.
# This makes it easy to swap SQLite for PostgreSQL or MongoDB in the future
# by simply writing a new repository class that implements the same functions.
# ==========================================

class PredictionRepository:
    @staticmethod
    def add_prediction(email_text: str, pred: PredictionResult) -> int:
        """
        Saves a new scan prediction result into the SQLite database.
        
        Returns:
            The integer ID of the newly inserted record.
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        cursor.execute(
            """
            INSERT INTO prediction_history (
                email_text, is_spam, probability, confidence, 
                model_used, vectorizer_used, execution_time_ms, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                email_text,
                1 if pred.is_spam else 0,
                pred.spam_probability,
                pred.confidence_score,
                pred.model_name,
                pred.vectorizer_name,
                pred.prediction_time_ms,
                timestamp
            )
        )
        
        conn.commit()
        last_id = cursor.lastrowid
        conn.close()
        
        logger.info(f"Logged prediction to database with ID: {last_id}")
        return last_id

    @staticmethod
    def get_history(
        limit: int = 100,
        search_query: Optional[str] = None,
        label_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves prediction history records from the database.
        Supports search filter (matching keywords in email text) and label filtering.
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = "SELECT * FROM prediction_history WHERE 1=1"
        params = []
        
        # Apply search filter
        if search_query:
            query += " AND email_text LIKE ?"
            params.append(f"%{search_query}%")
            
        # Apply label filter
        if label_filter:
            if label_filter.lower() == "spam":
                query += " AND is_spam = 1"
            elif label_filter.lower() == "ham":
                query += " AND is_spam = 0"
                
        # Order by newest first
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        # Convert sqlite3.Row objects to standard dictionaries
        history = [dict(row) for row in rows]
        return history

    @staticmethod
    def get_aggregate_stats() -> Dict[str, Any]:
        """
        Computes aggregate statistics of the history of scans.
        Used to display KPIs and charts on the user dashboard.
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Total counts, spam counts, averages
        cursor.execute(
            """
            SELECT 
                COUNT(*) as total_scans,
                SUM(CASE WHEN is_spam = 1 THEN 1 ELSE 0 END) as total_spam,
                SUM(CASE WHEN is_spam = 0 THEN 1 ELSE 0 END) as total_ham,
                AVG(confidence) as avg_confidence,
                AVG(execution_time_ms) as avg_latency
            FROM prediction_history
            """
        )
        summary_row = dict(cursor.fetchone())
        
        total = summary_row.get("total_scans") or 0
        
        if total == 0:
            conn.close()
            return {
                "total_scans": 0,
                "total_spam": 0,
                "total_ham": 0,
                "spam_percentage": 0.0,
                "avg_confidence": 0.0,
                "avg_latency": 0.0,
                "model_stats": {},
                "history_scans": []
            }
            
        # Stats by model
        cursor.execute(
            """
            SELECT model_used, COUNT(*) as count 
            FROM prediction_history 
            GROUP BY model_used
            """
        )
        model_rows = cursor.fetchall()
        model_stats = {row["model_used"]: row["count"] for row in model_rows}
        
        # Predictions over time (limit to last 30 scans for simple trends)
        cursor.execute(
            """
            SELECT id, is_spam, confidence, timestamp 
            FROM prediction_history 
            ORDER BY id ASC LIMIT 50
            """
        )
        time_rows = cursor.fetchall()
        time_stats = [dict(row) for row in time_rows]
        
        conn.close()
        
        spam_count = summary_row.get("total_spam") or 0
        spam_percentage = (spam_count / total * 100) if total > 0 else 0.0
        
        return {
            "total_scans": total,
            "total_spam": spam_count,
            "total_ham": summary_row.get("total_ham") or 0,
            "spam_percentage": round(spam_percentage, 2),
            "avg_confidence": round(summary_row.get("avg_confidence") or 0.0, 2),
            "avg_latency": round(summary_row.get("avg_latency") or 0.0, 2),
            "model_stats": model_stats,
            "history_scans": time_stats
        }

    @staticmethod
    def clear_history() -> None:
        """
        Deletes all historical records from the table.
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM prediction_history")
        conn.commit()
        conn.close()
        logger.info("Database prediction history cleared.")
