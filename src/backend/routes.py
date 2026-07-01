import logging
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Optional

from src.config import MODEL_DIR
from src.backend.schemas import (
    EmailInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    BatchItemResponse,
    HistoryRecord,
    SystemStatsResponse
)
from src.ml_engine.models import SpamClassifier
from src.database.repository import PredictionRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

@router.get("/health")
def health_check():
    """
    API health check endpoint.
    """
    return {"status": "online"}

# Global active classifier variable
_classifier: Optional[SpamClassifier] = None

def get_classifier() -> SpamClassifier:
    """
    Dependency helper to load and retrieve the active Spam Classifier.
    If the models have not been trained yet, it automatically triggers training first.
    """
    global _classifier
    if _classifier is not None:
        return _classifier

    metadata_path = MODEL_DIR / "metadata.json"

    # If the project is clean and models aren't trained, let's train them automatically!
    if not metadata_path.exists():
        logger.info("Metadata file not found. Auto-triggering model training pipeline...")
        from src.ml_engine.trainer import ModelTrainer
        trainer = ModelTrainer()
        trainer.train_and_compare()

    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        active = metadata["active_model"]
        model_name = active["model_name"]
        vec_name = active["vectorizer_name"]
        
        model_path = MODEL_DIR / active["model_filename"]
        vec_path = MODEL_DIR / active["vectorizer_filename"]

        # Instantiate our prediction classifier wrapper
        _classifier = SpamClassifier(
            model_path=str(model_path),
            vectorizer_path=str(vec_path),
            model_name=model_name,
            vectorizer_name=vec_name
        )
        return _classifier
    except Exception as e:
        logger.error(f"Failed to load the active classifier: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Machine learning engine is not ready: {str(e)}"
        )


@router.post("/analyze", response_model=PredictionResponse)
def analyze_email(payload: EmailInput):
    """
    Analyzes a single raw email message using the active machine learning model.
    Logs the result to the history database.
    """
    # 1. Get the trained classifier
    classifier = get_classifier()

    try:
        # 2. Run prediction
        result = classifier.predict(payload.text)

        # 3. Log the prediction to the database history
        PredictionRepository.add_prediction(payload.text, result)

        # 4. Return standard response
        return PredictionResponse(
            is_spam=result.is_spam,
            spam_probability=result.spam_probability,
            confidence_score=result.confidence_score,
            model_name=result.model_name,
            vectorizer_name=result.vectorizer_name,
            prediction_time_ms=result.prediction_time_ms
        )
    except Exception as e:
        logger.error(f"Error during email analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict-batch", response_model=BatchPredictionResponse)
def analyze_batch(payload: BatchPredictionRequest):
    """
    Scans a batch list of emails, computes overall statistics,
    and logs all results into the SQLite prediction history database.
    """
    classifier = get_classifier()
    
    predictions = []
    spam_count = 0
    
    try:
        for item in payload.emails:
            # Predict for each item
            result = classifier.predict(item.text)
            
            # Log individual record to database
            PredictionRepository.add_prediction(item.text, result)
            
            is_spam = result.is_spam
            if is_spam:
                spam_count += 1
                
            predictions.append(
                BatchItemResponse(
                    id=item.id,
                    text=item.text,
                    is_spam=is_spam,
                    spam_probability=result.spam_probability,
                    confidence_score=result.confidence_score
                )
            )
            
        total = len(payload.emails)
        spam_percentage = (spam_count / total * 100) if total > 0 else 0.0
        
        return BatchPredictionResponse(
            total_processed=total,
            spam_count=spam_count,
            ham_count=total - spam_count,
            spam_percentage=round(spam_percentage, 2),
            predictions=predictions
        )
    except Exception as e:
        logger.error(f"Error during batch analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@router.get("/models/compare")
def get_model_comparison():
    """
    Returns the evaluation reports for all trained models and vectorizers,
    along with indicating which model is currently 'active' (best performing).
    """
    metadata_path = MODEL_DIR / "metadata.json"
    if not metadata_path.exists():
        # Trigger training to generate comparison data if missing
        from src.ml_engine.trainer import ModelTrainer
        trainer = ModelTrainer()
        trainer.train_and_compare()

    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        logger.error(f"Error reading model comparison metadata: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read comparisons: {str(e)}")


def background_train_task():
    """
    Helper function run by background tasks to retrain the models.
    Refreshes the global classifier instance once complete.
    """
    global _classifier
    logger.info("Background retraining started.")
    try:
        from src.ml_engine.trainer import ModelTrainer
        trainer = ModelTrainer()
        trainer.train_and_compare()
        # Reset classifier to force reloading the newly trained active model
        _classifier = None
        logger.info("Background retraining completed. Classifier reset successfully.")
    except Exception as e:
        logger.error(f"Background retraining failed: {e}")


@router.post("/models/train")
def train_models(background_tasks: BackgroundTasks):
    """
    Triggers the machine learning pipeline to re-evaluate and re-train all models.
    Runs asynchronously in the background so the API does not block or timeout.
    """
    background_tasks.add_task(background_train_task)
    return {"status": "success", "message": "Model training pipeline triggered in background."}


@router.get("/history", response_model=List[HistoryRecord])
def get_history(
    limit: int = 100,
    search: Optional[str] = None,
    label: Optional[str] = None
):
    """
    Fetches prediction history records from the database.
    Allows filtering by label (spam/ham) and searching email text.
    """
    try:
        history = PredictionRepository.get_history(limit=limit, search_query=search, label_filter=label)
        return history
    except Exception as e:
        logger.error(f"Failed to fetch history: {e}")
        raise HTTPException(status_code=500, detail=f"Database fetch failed: {str(e)}")


@router.get("/history/stats", response_model=SystemStatsResponse)
def get_history_stats():
    """
    Returns aggregate statistics (total queries, spam rates, average confidence)
    from prediction logs stored in SQLite.
    """
    try:
        stats = PredictionRepository.get_aggregate_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to calculate database statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Stats calculation failed: {str(e)}")


@router.post("/history/clear")
def clear_history():
    """
    Deletes all historical records from the SQLite database.
    """
    try:
        PredictionRepository.clear_history()
        return {"status": "success", "message": "Prediction history database cleared successfully."}
    except Exception as e:
        logger.error(f"Failed to clear history: {e}")
        raise HTTPException(status_code=500, detail=f"Database clear failed: {str(e)}")
