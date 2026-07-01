from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

# ==========================================
# DESIGN DECISION: Domain Entities (Clean Architecture)
# These dataclasses represent the core business objects of our application.
# They are completely independent of any framework, database, or ML library.
# By using native Python dataclasses, we keep the business logic clean and separable.
# ==========================================

@dataclass
class EmailMessage:
    """
    Represents a single email message inside our system.
    
    Attributes:
        text (str): The body text of the email.
        label (Optional[str]): Whether it is 'spam' or 'ham' (clean). None if we don't know yet.
        id (Optional[int]): The database ID (if stored in a database).
        timestamp (datetime): The date and time when this email was scanned.
    """
    text: str
    label: Optional[str] = None
    id: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PredictionResult:
    """
    Represents the output from our ML models after scanning an email.
    
    Attributes:
        is_spam (bool): True if the model thinks the email is spam, False if it is clean.
        spam_probability (float): How likely the email is spam (from 0.0 to 1.0).
        confidence_score (float): A percentage score (0% to 100%) indicating how confident the model is.
        model_name (str): The name of the AI model used (e.g., 'Random Forest').
        vectorizer_name (str): The text-to-numbers method used (e.g., 'TF-IDF' or 'Word2Vec').
        prediction_time_ms: How long the prediction took (in milliseconds).
    """
    is_spam: bool
    spam_probability: float
    confidence_score: float
    model_name: str
    vectorizer_name: str
    prediction_time_ms: float


@dataclass
class ModelMetrics:
    """
    Represents the performance card of a trained machine learning model.
    
    Attributes:
        model_name (str): The name of the machine learning model.
        vectorizer_name (str): The vectorization technique (TF-IDF vs Word2Vec).
        accuracy (float): Percentage of correct guesses out of all emails.
        precision (float): Out of all emails guessed as spam, how many were actually spam.
        recall (float): Out of all actual spam emails, how many did we successfully catch.
        f1_score (float): The balanced score between precision and recall.
        confusion_matrix (List[List[int]]): A 2x2 table tracking [TN, FP, FN, TP] counts.
    """
    model_name: str
    vectorizer_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]
