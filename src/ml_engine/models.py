import logging
import time
from typing import Union
import joblib

from src.core.entities import PredictionResult
from src.data_pipeline.preprocessor import EmailPreprocessor
from src.ml_engine.vectorizers import TFIDFVectorizerWrapper, Word2VecVectorizerWrapper

logger = logging.getLogger(__name__)

# ==========================================
# DESIGN DECISION: Unified Prediction Wrapper
# To run predictions in our FastAPI backend without exposing raw ML details,
# we wrap the active model and active vectorizer inside a single 'SpamClassifier' class.
# This class acts as the interface, automating the clean-vectorize-predict pipeline
# and returning the structured 'PredictionResult' entity.
# ==========================================

class SpamClassifier:
    def __init__(
        self,
        model_path: str,
        vectorizer_path: str,
        model_name: str,
        vectorizer_name: str
    ):
        """
        Loads a trained model and vectorizer from the disk to prepare for prediction.
        
        Args:
            model_path (str): Filepath to the serialized model (.pkl).
            vectorizer_path (str): Filepath to the serialized vectorizer (.pkl or .bin).
            model_name (str): The name of the algorithm (e.g. 'Random Forest').
            vectorizer_name (str): The vectorization type (e.g. 'TF-IDF' or 'Word2Vec').
        """
        self.model_name = model_name
        self.vectorizer_name = vectorizer_name
        self.preprocessor = EmailPreprocessor()

        # Load the model
        logger.info(f"Loading model '{model_name}' from: {model_path}")
        self.model = joblib.load(model_path)

        # Load the vectorizer
        logger.info(f"Loading vectorizer '{vectorizer_name}' from: {vectorizer_path}")
        if vectorizer_name == "TF-IDF":
            self.vectorizer = TFIDFVectorizerWrapper()
            self.vectorizer.load(vectorizer_path)
        elif vectorizer_name == "Word2Vec":
            self.vectorizer = Word2VecVectorizerWrapper()
            self.vectorizer.load(vectorizer_path)
        else:
            raise ValueError(f"Unknown vectorizer type: {vectorizer_name}")

    def predict(self, raw_text: str) -> PredictionResult:
        """
        Runs the full prediction pipeline on raw text:
        1. Preprocesses the text (lowercasing, cleaning, lemmatizing).
        2. Vectorizes the text into a numerical format.
        3. Predicts the class and probability.
        4. Calculates a confidence score.
        5. Measures performance time in milliseconds.
        """
        start_time = time.time()

        # 1. Clean the text
        clean_text = self.preprocessor.preprocess(raw_text)

        # Handle edge case: if the email became completely empty after cleaning
        if not clean_text:
            return PredictionResult(
                is_spam=False,
                spam_probability=0.0,
                confidence_score=100.0,
                model_name=self.model_name,
                vectorizer_name=self.vectorizer_name,
                prediction_time_ms=0.0
            )

        # 2. Vectorize the text
        # The vectorizer expects a list of inputs, so we wrap it in a list [clean_text]
        vectorized_text = self.vectorizer.transform([clean_text])

        # 3. Predict Class and Probability
        # Most models in scikit-learn support predict_proba, which returns [probability_ham, probability_spam]
        # In case a model does not support it (like SVM without probability=True), we fall back
        try:
            probabilities = self.model.predict_proba(vectorized_text)[0]
            # Index 1 corresponds to the positive class ('spam')
            spam_probability = float(probabilities[1])
        except AttributeError:
            # Fallback for models without predict_proba (like a raw SVM)
            prediction = self.model.predict(vectorized_text)[0]
            spam_probability = 1.0 if prediction == "spam" else 0.0

        is_spam = spam_probability >= 0.5

        # 4. Calculate Confidence Score
        # DESIGN DECISION: Confidence Metric
        # If probability is 0.5, we are 50/50 (0% confident).
        # If probability is 0.0 (certainly ham) or 1.0 (certainly spam), we are 100% confident.
        # Formula: confidence = abs(probability - 0.5) * 2
        confidence = abs(spam_probability - 0.5) * 2.0
        confidence_percentage = round(confidence * 100.0, 2)

        # 5. Stop the clock
        elapsed_time_ms = (time.time() - start_time) * 1000.0

        return PredictionResult(
            is_spam=is_spam,
            spam_probability=round(spam_probability, 4),
            confidence_score=confidence_percentage,
            model_name=self.model_name,
            vectorizer_name=self.vectorizer_name,
            prediction_time_ms=round(elapsed_time_ms, 2)
        )
