from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# ==========================================
# DESIGN DECISION: Pydantic Schema Validation
# Standardizing API inputs and outputs with Pydantic ensures:
# 1. Automatic data validation and sanitation (prevents malicious inputs).
# 2. Rich API documentation (interactive Swagger UI /docs automatically generated).
# 3. Clean mapping between our core dataclass Entities and JSON format.
# ==========================================

class EmailInput(BaseModel):
    """
    Validation schema for single email scanning request.
    """
    text: str = Field(..., min_length=1, description="The raw content of the email to analyze.")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Dear customer, your card ending in 4321 is blocked. Please click http://scam.link to unlock."
            }
        }


class PredictionResponse(BaseModel):
    """
    Validation schema for single prediction output.
    """
    is_spam: bool = Field(..., description="True if email is spam, False if ham.")
    spam_probability: float = Field(..., description="Probability of being spam (0.0 to 1.0).")
    confidence_score: float = Field(..., description="Model certainty score (0% to 100%).")
    model_name: str = Field(..., description="Name of the ML model used.")
    vectorizer_name: str = Field(..., description="Name of the vectorizer used.")
    prediction_time_ms: float = Field(..., description="Time taken to make prediction in milliseconds.")


class EmailBatchItem(BaseModel):
    """
    A single email item inside a batch request.
    """
    id: Optional[str] = Field(None, description="Optional ID for tracking specific emails.")
    text: str = Field(..., min_length=1, description="The raw email text.")


class BatchPredictionRequest(BaseModel):
    """
    Request schema for bulk email predictions.
    """
    emails: List[EmailBatchItem] = Field(..., min_length=1, description="List of emails to classify in bulk.")


class BatchItemResponse(BaseModel):
    """
    Result of a single email prediction inside a batch response.
    """
    id: Optional[str] = Field(None, description="The optional ID passed in the request.")
    text: str = Field(..., description="The raw email text that was analyzed.")
    is_spam: bool = Field(..., description="True if spam, False if ham.")
    spam_probability: float = Field(..., description="Probability score.")
    confidence_score: float = Field(..., description="Confidence percentage.")


class BatchPredictionResponse(BaseModel):
    """
    Response schema for bulk predictions, including aggregations.
    """
    total_processed: int = Field(..., description="Total number of emails classified.")
    spam_count: int = Field(..., description="Total spam emails detected.")
    ham_count: int = Field(..., description="Total clean emails detected.")
    spam_percentage: float = Field(..., description="Percentage of spam emails (0.0 to 100.0).")
    predictions: List[BatchItemResponse] = Field(..., description="Individual classification results.")


class HistoryRecord(BaseModel):
    """
    Schema representing a single database log record in history list.
    """
    id: int
    email_text: str
    is_spam: bool
    probability: float
    confidence: float
    model_used: str
    vectorizer_used: str
    execution_time_ms: float
    timestamp: str


class SystemStatsResponse(BaseModel):
    """
    Schema for history aggregate statistics on the dashboard.
    """
    total_scans: int
    total_spam: int
    total_ham: int
    spam_percentage: float
    avg_confidence: float
    avg_latency: float
    model_stats: Dict[str, int]
    history_scans: List[Dict[str, Any]]
