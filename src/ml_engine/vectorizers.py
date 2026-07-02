import logging
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# ==========================================
# DESIGN DECISION: Unified Vectorizer Interface
# Computers do not understand words; they only understand numbers.
# We create a clean wrapper for:
# 1. TF-IDF: Counts word importance based on frequencies.
# 2. Word2Vec: Maps words to a dense continuous vector space where words
#    with similar meanings (like 'cash' and 'money') sit close together.
# To make Word2Vec compatible with standard Scikit-Learn models, we write a
# custom wrapper that implements Scikit-Learn's standard fit/transform API.
# ==========================================

class TFIDFVectorizerWrapper:
    def __init__(self, max_features=5000):
        """
        TF-IDF counts how often words appear, but penalizes extremely common words (like 'the').
        We use bi-grams and sublinear term frequency to capture multi-word combinations.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.name = "TF-IDF"

    def fit(self, X):
        """
        Learns the vocabulary (the list of all words) from the training emails.
        """
        self.vectorizer.fit(X)
        return self

    def transform(self, X):
        """
        Converts the list of emails into a table of numbers (matrix) using TF-IDF weights.
        """
        # Converts sparse matrix returned by TF-IDF to a dense numpy array for models
        return self.vectorizer.transform(X).toarray()

    def fit_transform(self, X):
        """
        Learns the vocabulary and converts the training emails in one quick step.
        """
        return self.vectorizer.fit_transform(X).toarray()

    def save(self, filepath):
        """
        Saves the trained TF-IDF rules to a file on disk so we can reuse it later.
        """
        joblib.dump(self.vectorizer, filepath)
        logger.info(f"TF-IDF Vectorizer saved to {filepath}")

    def load(self, filepath):
        """
        Loads the saved TF-IDF rules from the disk.
        """
        self.vectorizer = joblib.load(filepath)
        logger.info(f"TF-IDF Vectorizer loaded from {filepath}")
        return self


class Word2VecVectorizerWrapper:
    def __init__(self, vector_size=100, window=5, min_count=1):
        """
        Word2Vec translates words into coordinates (like maps).
        
        Args:
            vector_size (int): The number of dimensions for each word (like 100 coordinates).
            window (int): How many words nearby the target word the algorithm looks at.
            min_count (int): Ignores words that appear less than this many times.
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.w2v_model = None
        self.name = "Word2Vec"

    def _tokenize_emails(self, X):
        """
        Helper function to turn an array of text sentences into lists of words.
        """
        return [str(text).split() for text in X]

    def fit(self, X):
        """
        Trains the custom Word2Vec neural embedding map on our email dataset.
        It learns to group similar words together in a 100-dimensional space.
        """
        from gensim.models import Word2Vec
        tokenized_emails = self._tokenize_emails(X)
        logger.info("Training custom Word2Vec model...")
        self.w2v_model = Word2Vec(
            sentences=tokenized_emails,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4,
            epochs=10
        )
        return self

    def _get_mean_vector(self, words):
        """
        A single email has many words, each having a 100-dimensional coordinate.
        This function averages all the coordinates together to find the
        'center point' vector representing the entire email message.
        """
        # Get vectors of words that actually exist in our trained Word2Vec model dictionary
        valid_vectors = [self.w2v_model.wv[word] for word in words if word in self.w2v_model.wv]
        
        if not valid_vectors:
            # If no words in the email were learned, return a vector of zeros
            return np.zeros(self.vector_size)
            
        return np.mean(valid_vectors, axis=0)

    def transform(self, X):
        """
        Translates a list of emails into a table of averaged Word2Vec vectors.
        """
        if self.w2v_model is None:
            raise ValueError("Word2Vec model has not been trained yet. Call fit first.")
            
        tokenized_emails = self._tokenize_emails(X)
        features = [self._get_mean_vector(email) for email in tokenized_emails]
        return np.array(features)

    def fit_transform(self, X):
        """
        Trains the Word2Vec model and converts the emails in one step.
        """
        return self.fit(X).transform(X)

    def save(self, filepath):
        """
        Saves the Word2Vec model to disk.
        Since gensim models can't be easily pickled with joblib, we use gensim's native save method.
        """
        if self.w2v_model is not None:
            self.w2v_model.save(str(filepath))
            logger.info(f"Word2Vec model saved to {filepath}")
        else:
            logger.warning("No Word2Vec model to save.")

    def load(self, filepath):
        """
        Loads the Word2Vec model from disk.
        """
        from gensim.models import Word2Vec
        self.w2v_model = Word2Vec.load(str(filepath))
        logger.info(f"Word2Vec model loaded from {filepath}")
        return self
