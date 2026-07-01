import logging
from typing import List
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from src.core.entities import ModelMetrics

logger = logging.getLogger(__name__)

# ==========================================
# DESIGN DECISION: Unified Evaluator
# This class calculates all standard performance metrics for a classification model.
# By returning our core 'ModelMetrics' entity, it decouples the evaluation metrics
# calculations from the training pipeline and backend routes.
# ==========================================

class ModelEvaluator:
    @staticmethod
    def evaluate(y_true: List[str], y_pred: List[str], model_name: str, vectorizer_name: str) -> ModelMetrics:
        """
        Compares the actual correct labels against the model's guesses.
        Calculates and returns standard machine learning metrics:
        - Accuracy: Overall correct rate.
        - Precision: Rate of true spam out of predicted spam.
        - Recall: Rate of true spam caught.
        - F1 Score: Balanced metric combining Precision and Recall.
        - Confusion Matrix: Grid of [TN, FP, FN, TP].
        """
        # Calculate standard metrics
        # pos_label="spam" tells scikit-learn that "spam" is the class we want to detect.
        # zero_division=0 handles cases where a model predicts zero spam, avoiding division-by-zero crashes.
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label="spam", zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label="spam", zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label="spam", zero_division=0)
        
        # Calculate the 2x2 Confusion Matrix:
        # [ [True Negatives, False Positives],
        #   [False Negatives, True Positives] ]
        cm = confusion_matrix(y_true, y_pred, labels=["ham", "spam"])
        cm_list = cm.tolist()  # Convert numpy 2D array to a standard Python list of lists

        logger.info(
            f"Evaluated model '{model_name}' + '{vectorizer_name}': "
            f"Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}"
        )

        return ModelMetrics(
            model_name=model_name,
            vectorizer_name=vectorizer_name,
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
            confusion_matrix=cm_list
        )
