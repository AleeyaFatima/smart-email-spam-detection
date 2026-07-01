import logging
import requests
from typing import List, Dict, Any, Optional

from src.config import BACKEND_URL

logger = logging.getLogger(__name__)

# ==========================================
# DESIGN DECISION: API Client (Service Layer)
# The dashboard frontend should never make raw HTTP calls directly.
# By centralizing all backend communication inside this 'APIClient' service,
# we ensure consistent error handling, request timeouts, and clean interfaces.
# If the API endpoints ever change, we only update this file.
# ==========================================

class APIClient:
    def __init__(self, base_url: str = BACKEND_URL):
        self.base_url = base_url

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Helper method to perform GET requests with safety error handling.
        """
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"GET request to {url} failed: {e}")
            raise ConnectionError(f"Cannot connect to Backend API. Please ensure the server is running on {self.base_url}.")

    def _post(self, endpoint: str, json_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Helper method to perform POST requests with safety error handling.
        """
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.post(url, json=json_data, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"POST request to {url} failed: {e}")
            raise ConnectionError(f"Cannot connect to Backend API. Please ensure the server is running on {self.base_url}.")

    def check_health(self) -> bool:
        """
        Checks if the backend REST API is online.
        """
        try:
            response = requests.get(f"{self.base_url}/", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def analyze_email(self, text: str) -> Dict[str, Any]:
        """
        Sends a single email body for spam analysis.
        """
        return self._post("/api/analyze", json_data={"text": text})

    def analyze_batch(self, emails: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Sends a batch list of emails for bulk spam analysis.
        
        Args:
            emails: A list of dicts, e.g. [{"id": "1", "text": "email body"}, ...]
        """
        return self._post("/api/predict-batch", json_data={"emails": emails})

    def get_history(self, limit: int = 100, search: Optional[str] = None, label: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves prediction log history.
        """
        params = {"limit": limit}
        if search:
            params["search"] = search
        if label and label != "All":
            params["label"] = label.lower()
        return self._get("/api/history", params=params)

    def get_stats(self) -> Dict[str, Any]:
        """
        Retrieves aggregated database scan statistics (KPIs).
        """
        return self._get("/api/history/stats")

    def get_model_comparison(self) -> Dict[str, Any]:
        """
        Retrieves performance metrics for all trained models.
        """
        return self._get("/api/models/compare")

    def retrain_models(self) -> Dict[str, Any]:
        """
        Triggers background model retraining pipeline.
        """
        return self._post("/api/models/train")

    def clear_history(self) -> Dict[str, Any]:
        """
        Deletes all history logs in the SQLite database.
        """
        return self._post("/api/history/clear")
