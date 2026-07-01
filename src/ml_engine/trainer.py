import json
import logging
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from src.config import MODEL_DIR
from src.data_pipeline.loader import DataLoader
from src.ml_engine.vectorizers import TFIDFVectorizerWrapper, Word2VecVectorizerWrapper
from src.ml_engine.evaluator import ModelEvaluator

logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ==========================================
# DESIGN DECISION: Production-Grade Model Trainer
# This pipeline automates training and comparing 8 different combinations:
# 4 models (Logistic Regression, Naive Bayes, SVM, Random Forest)
# cross-analyzed over 2 feature types (TF-IDF and Word2Vec).
# It evaluates them all, saves the binaries, and writes a 'metadata.json' file
# tracking performance and designating the 'best' model based on F1 Score.
# ==========================================

class ModelTrainer:
    def __init__(self):
        """
        Initializes the trainer. Prepares the data loader.
        """
        self.loader = DataLoader()

    def train_and_compare(self) -> dict:
        """
        Main pipeline function to train and compare all models.
        """
        logger.info("Starting training pipeline...")

        # 1. Load data and split into train/test
        X_train, X_test, y_train, y_test = self.loader.get_splits()
        logger.info(f"Loaded dataset splits. Train size: {len(X_train)}, Test size: {len(X_test)}")

        # 2. Fit and initialize vectorizers
        tfidf_wrapper = TFIDFVectorizerWrapper().fit(X_train)
        w2v_wrapper = Word2VecVectorizerWrapper().fit(X_train)

        # Convert text splits to numerical arrays
        X_train_tfidf = tfidf_wrapper.transform(X_train)
        X_test_tfidf = tfidf_wrapper.transform(X_test)

        X_train_w2v = w2v_wrapper.transform(X_train)
        X_test_w2v = w2v_wrapper.transform(X_test)

        # Save both vectorizers
        tfidf_path = MODEL_DIR / "tfidf_vectorizer.pkl"
        w2v_path = MODEL_DIR / "word2vec_model.bin"
        tfidf_wrapper.save(tfidf_path)
        w2v_wrapper.save(w2v_path)

        # 3. Define the models to train
        # MultinomialNB requires non-negative values, so it's perfect for TF-IDF.
        # GaussianNB handles dense real-valued coordinates (like Word2Vec) beautifully.
        models_config = {
            "TF-IDF": {
                "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
                "Naive Bayes": MultinomialNB(),
                "Support Vector Machine": SVC(random_state=42, probability=True),
                "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100)
            },
            "Word2Vec": {
                "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
                "Naive Bayes": GaussianNB(),
                "Support Vector Machine": SVC(random_state=42, probability=True),
                "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100)
            }
        }

        # Dictionary to store performance cards for all models
        evaluation_results = []
        best_f1 = -1.0
        best_model_info = {}

        # 4. Train, predict, evaluate, and save each model
        for vec_name, models_dict in models_config.items():
            # Select the correct vectorized data based on vectorizer type
            X_tr = X_train_tfidf if vec_name == "TF-IDF" else X_train_w2v
            X_te = X_test_tfidf if vec_name == "TF-IDF" else X_test_w2v
            vec_path = tfidf_path if vec_name == "TF-IDF" else w2v_path

            for model_name, model in models_dict.items():
                logger.info(f"Training {model_name} using {vec_name}...")
                model.fit(X_tr, y_train)

                # Predict on test split
                y_pred = model.predict(X_te)

                # Evaluate performance using our ModelEvaluator
                metrics = ModelEvaluator.evaluate(y_test, y_pred, model_name, vec_name)

                # Create file name to save the model weights
                # e.g. "logistic_regression_tfidf.pkl"
                sanitized_model_name = model_name.lower().replace(" ", "_")
                sanitized_vec_name = vec_name.lower().replace("-", "_")
                model_filename = f"{sanitized_model_name}_{sanitized_vec_name}.pkl"
                model_save_path = MODEL_DIR / model_filename

                # Save the trained model weights to disk
                joblib.dump(model, model_save_path)

                # Store metrics list for metadata JSON
                metrics_dict = {
                    "model_name": metrics.model_name,
                    "vectorizer_name": metrics.vectorizer_name,
                    "accuracy": metrics.accuracy,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1_score": metrics.f1_score,
                    "confusion_matrix": metrics.confusion_matrix,
                    "model_filename": model_filename,
                    "vectorizer_filename": vec_path.name
                }
                evaluation_results.append(metrics_dict)

                # Check if this model is the best performer based on F1 Score
                # DESIGN DECISION: F1 Score is the best choice for selecting the active model
                # because email databases are highly imbalanced (far more ham than spam).
                # F1 Score balances Precision (carefulness) and Recall (catching rate).
                if metrics.f1_score > best_f1:
                    best_f1 = metrics.f1_score
                    best_model_info = {
                        "model_name": model_name,
                        "vectorizer_name": vec_name,
                        "f1_score": metrics.f1_score,
                        "model_filename": model_filename,
                        "vectorizer_filename": vec_path.name
                    }

        # 5. Save the training metadata and active model pointers to a metadata.json file
        metadata = {
            "active_model": best_model_info,
            "all_models": evaluation_results,
            "dataset_stats": self.loader.get_stats()
        }

        metadata_path = MODEL_DIR / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Model comparison metadata saved to {metadata_path}")
        logger.info(f"Active Model Chosen: {best_model_info['model_name']} with {best_model_info['vectorizer_name']} (F1: {best_model_info['f1_score']:.4f})")

        return metadata

if __name__ == "__main__":
    # Let's run training if executed directly
    trainer = ModelTrainer()
    trainer.train_and_compare()
