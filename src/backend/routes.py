import logging
import json
import shutil
import tempfile
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from typing import List, Optional

from src.config import MODEL_DIR, RAW_DATA_PATH, PROCESSED_DATA_PATH
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


@router.post("/dataset/upload")
def upload_dataset(file: UploadFile = File(...), mode: str = "overwrite"):
    """
    Uploads a new training dataset (CSV or TSV format).
    mode can be:
    - 'overwrite': replaces the current spam_dataset.csv
    - 'append': appends new rows to the current spam_dataset.csv
    """
    try:
        import pandas as pd
        
        # Save uploaded file to a temporary file first to validate it
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        temp_path = Path(temp_file.name)
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Try parsing it to check if it's valid
        # We read first few lines to check delimiter
        try:
            with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline()
            sep = "\t" if "\t" in first_line else ","
            
            # Use raw column check
            df_new = pd.read_csv(temp_path, sep=sep, encoding="utf-8", on_bad_lines="skip")
            
            # Check columns
            cols = [c.lower().strip() for c in df_new.columns]
            label_candidates = {"label", "v1", "class", "target", "spam", "category"}
            text_candidates = {"text", "v2", "message", "email", "body", "sms"}
            
            label_col = None
            text_col = None
            for idx, col in enumerate(cols):
                if col in label_candidates:
                    label_col = df_new.columns[idx]
                elif col in text_candidates:
                    text_col = df_new.columns[idx]
                    
            if label_col is None or text_col is None:
                # Let's check if we can read it without headers
                df_no_header = pd.read_csv(temp_path, sep=sep, header=None, encoding="utf-8", on_bad_lines="skip")
                if len(df_no_header.columns) >= 2:
                    # Use first column as label, second as text
                    df_new = df_no_header.rename(columns={0: "label", 1: "text"})
                else:
                    raise ValueError("Uploaded file must have at least 2 columns: label and text content.")
            else:
                df_new = df_new.rename(columns={label_col: "label", text_col: "text"})
                
            df_new = df_new[["label", "text"]]
            
        except Exception as parse_err:
            try:
                temp_path.unlink()
            except OSError:
                pass
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV/TSV format: {str(parse_err)}")
        
        # Standardize labels
        df_new["label"] = df_new["label"].astype(str).str.lower().str.strip()
        df_new["label"] = df_new["label"].replace({
            "1": "spam", "0": "ham", 
            "1.0": "spam", "0.0": "ham",
            "yes": "spam", "no": "ham",
            "true": "spam", "false": "ham",
            "positive": "spam", "negative": "ham"
        })
        df_new = df_new[df_new["label"].isin(["spam", "ham"])]
        
        if len(df_new) == 0:
            try:
                temp_path.unlink()
            except OSError:
                pass
            raise HTTPException(status_code=400, detail="No valid rows containing 'spam' or 'ham' labels were found.")
            
        # Append or Overwrite
        if mode == "append" and RAW_DATA_PATH.exists():
            try:
                # Read existing with separator detection
                with open(RAW_DATA_PATH, "r", encoding="utf-8", errors="ignore") as f:
                    ex_first = f.readline()
                ex_sep = "\t" if "\t" in ex_first else ","
                
                df_ex_check = pd.read_csv(RAW_DATA_PATH, sep=ex_sep, nrows=5, header=None)
                ex_first_row = [str(val).lower().strip() for val in df_ex_check.iloc[0]]
                ex_has_header = any(col in label_candidates or col in text_candidates for col in ex_first_row)
                
                if ex_has_header:
                    df_existing = pd.read_csv(RAW_DATA_PATH, sep=ex_sep, encoding="utf-8", on_bad_lines="skip")
                    rename_dict = {}
                    for idx, col in enumerate(df_existing.columns):
                        c_low = col.lower().strip()
                        if c_low in label_candidates:
                            rename_dict[col] = "label"
                        elif c_low in text_candidates:
                            rename_dict[col] = "text"
                    df_existing = df_existing.rename(columns=rename_dict)[["label", "text"]]
                else:
                    df_existing = pd.read_csv(RAW_DATA_PATH, sep=ex_sep, names=["label", "text"], header=None, encoding="utf-8", on_bad_lines="skip")
                    df_existing = df_existing[["label", "text"]]
                    
                df_merged = pd.concat([df_existing, df_new], ignore_index=True)
            except Exception as ex_err:
                logger.warning(f"Failed to load existing dataset for append: {ex_err}. Treating as overwrite.")
                df_merged = df_new
        else:
            df_merged = df_new
            
        # Save as tab-separated with no headers, to be consistent with raw format
        df_merged.dropna(subset=["label", "text"], inplace=True)
        df_merged.to_csv(RAW_DATA_PATH, sep="\t", index=False, header=False)
        
        # Clean up temp file
        try:
            temp_path.unlink()
        except OSError:
            pass
            
        # Remove processed dataset to force reloading
        if PROCESSED_DATA_PATH.exists():
            try:
                PROCESSED_DATA_PATH.unlink()
            except OSError:
                pass
            
        return {
            "status": "success",
            "message": f"Successfully uploaded dataset containing {len(df_new)} records. Total training pool is now {len(df_merged)} records.",
            "total_records": len(df_merged),
            "uploaded_records": len(df_new)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload training dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
