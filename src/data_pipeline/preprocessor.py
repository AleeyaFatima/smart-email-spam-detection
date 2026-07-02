import logging
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from src.config import NLTK_DATA_DIR

logger = logging.getLogger(__name__)

# ==========================================
# DESIGN DECISION: Robust and Portable Preprocessor
# Text preprocessing is the most important step in NLP.
# We download and configure NLTK locally inside the project folder (as set in config).
# This prevents permission errors on Windows and makes the code run offline
# after the first download.
# We use Lemmatization (which understands word meaning, e.g., 'mice' -> 'mouse')
# instead of Stemming (which just chops off word ends, e.g., 'arguing' -> 'argu').
# ==========================================

class EmailPreprocessor:
    def __init__(self):
        """
        Initializes the preprocessor. Automatically checks and downloads NLTK assets
        like the stopwords list, the tokenizer rules, and the WordNet lemmatizer dictionary.
        """
        # Ensure NLTK data downloads to our custom project folder
        nltk.data.path.append(str(NLTK_DATA_DIR))
        
        # Download necessary NLTK packages if they are not found locally
        for resource in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
            try:
                # We try to load it first; if it fails, we download it
                if resource == "stopwords":
                    nltk.data.find("corpora/stopwords")
                elif resource == "wordnet":
                    nltk.data.find("corpora/wordnet")
                elif resource == "punkt":
                    nltk.data.find("tokenizers/punkt")
                elif resource == "omw-1.4":
                    nltk.data.find("corpora/omw-1.4")
            except LookupError:
                logger.info(f"Downloading NLTK resource '{resource}'...")
                nltk.download(resource, download_dir=str(NLTK_DATA_DIR), quiet=True)

        self.lemmatizer = WordNetLemmatizer()
        # We load English stopwords (e.g., 'is', 'the', 'on') to filter them out later
        self.stop_words = set(stopwords.words("english"))

    def preprocess(self, text: str) -> str:
        """
        Cleans the email text to prepare it for the AI model.
        
        Steps:
        1. Convert text to lowercase.
        2. Remove HTML tags and web URLs.
        3. Remove numbers and punctuation marks.
        4. Split sentences into individual word tokens.
        5. Remove stopwords (common, meaningless words).
        6. Lemmatize words (reduce them to their root forms).
        7. Combine back into a single clean string.
        """
        if not text or not isinstance(text, str):
            return ""

        # 1. Convert to Lowercase
        text = text.lower()

        # 2. Clean HTML tags (e.g., <p> or <br/>)
        text = re.sub(r"<[^>]*>", " ", text)

        # 3. Clean URLs/Web Links (e.g., http://google.com)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        # 4. Clean Email Addresses (e.g., test@example.com)
        text = re.sub(r"\S+@\S+", " ", text)

        # 5. Preserve special markers for spam signals (currency, exclamation, numbers)
        text = re.sub(r"[\$\£\€]", " moneytoken ", text)
        text = re.sub(r"\!", " exclamationtoken ", text)
        text = re.sub(r"\d+", " numbertoken ", text)

        # 6. Remove Punctuation (e.g., !, ?, ., ,)
        # string.punctuation is a list of punctuation characters like '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        translator = str.maketrans("", "", string.punctuation)
        text = text.translate(translator)

        # 7. Tokenization (split text into words)
        try:
            tokens = word_tokenize(text)
        except Exception:
            # Fallback if tokenizer rules fails
            tokens = text.split()

        # 8. Stopwords Removal and Lemmatization
        clean_tokens = []
        for word in tokens:
            # Remove whitespace and filter out stop words and single characters
            word = word.strip()
            if word and word not in self.stop_words and len(word) > 1:
                # Reduce word to root form (e.g., "running" -> "run", "better" -> "good")
                root_word = self.lemmatizer.lemmatize(word)
                clean_tokens.append(root_word)

        # 9. Rejoin tokens back into a single clean sentence
        return " ".join(clean_tokens)

    def tokenize_clean_text(self, clean_text: str) -> list[str]:
        """
        Splits preprocessed clean text back into a list of words.
        This is helpful when training the Word2Vec model, which requires
        lists of words (sentences) instead of a single string.
        """
        return clean_text.split()
