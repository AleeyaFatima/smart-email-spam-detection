import os
import sys
import subprocess
import argparse
import time
import logging
from pathlib import Path

from src.config import BACKEND_HOST, BACKEND_PORT

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==========================================
# DESIGN DECISION: Robust Virtual Env Process Runner
# To avoid 'ModuleNotFoundError: No module named src' and package loading issues:
# 1. We run subprocesses using 'sys.executable' (the current active Python binary).
#    This guarantees they execute inside our active virtual environment (.venv).
# 2. We inject 'PYTHONPATH' pointing to the project root directory.
# 3. We run services as Python modules ('python -m uvicorn', 'python -m streamlit').
# This ensures cross-platform consistency without depending on global paths.
# ==========================================

# Define base directory
BASE_DIR = Path(__file__).resolve().parent

def get_run_env():
    """
    Prepares a clean copy of system environment variables with PYTHONPATH set.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BASE_DIR)
    return env


def run_training():
    """
    Launches the training and model comparison pipeline script.
    """
    logger.info("Starting model training pipeline...")
    cmd = [sys.executable, "src/ml_engine/trainer.py"]
    try:
        subprocess.run(cmd, env=get_run_env(), check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


def run_backend():
    """
    Launches the FastAPI backend server using Uvicorn module.
    """
    logger.info(f"Starting FastAPI backend server on http://{BACKEND_HOST}:{BACKEND_PORT}...")
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "src.backend.app:app", 
        "--host", BACKEND_HOST, 
        "--port", str(BACKEND_PORT),
        "--reload"
    ]
    try:
        subprocess.run(cmd, env=get_run_env())
    except KeyboardInterrupt:
        logger.info("FastAPI backend stopped by user.")


def run_frontend():
    """
    Launches the Streamlit dashboard app.
    """
    logger.info("Starting Streamlit Dashboard application...")
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        "frontend/app.py", 
        "--server.port", "8501", 
        "--server.address", "127.0.0.1"
    ]
    try:
        subprocess.run(cmd, env=get_run_env())
    except KeyboardInterrupt:
        logger.info("Streamlit Dashboard stopped by user.")


def run_all():
    """
    Launches both Uvicorn and Streamlit concurrently.
    Prints status and allows stopping both with a single Ctrl+C.
    """
    logger.info("Launching full stack: FastAPI Backend + Streamlit Dashboard...")
    env = get_run_env()
    
    # 1. Start FastAPI Backend process
    backend_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "src.backend.app:app", 
        "--host", BACKEND_HOST, 
        "--port", str(BACKEND_PORT)
    ], env=env)
    
    # Give the backend 4 seconds to spin up and load/download models
    logger.info("Waiting for FastAPI server to initialize...")
    time.sleep(4)
    
    # 2. Start Streamlit Dashboard process
    frontend_process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", 
        "frontend/app.py", 
        "--server.port", "8501", 
        "--server.address", "127.0.0.1"
    ], env=env)
    
    logger.info("Both servers are online!")
    logger.info(f"👉 FastAPI API Swagger: http://{BACKEND_HOST}:{BACKEND_PORT}/docs")
    logger.info("👉 Streamlit Dashboard: http://127.0.0.1:8501")
    logger.info("Press Ctrl+C to terminate both servers concurrently.")
    
    try:
        # Wait for both processes to run
        while True:
            # Check if any process terminated unexpectedly
            if backend_process.poll() is not None:
                logger.warning("Backend process terminated unexpectedly.")
                break
            if frontend_process.poll() is not None:
                logger.warning("Frontend process terminated unexpectedly.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down processes...")
    finally:
        # Gracefully terminate both processes
        backend_process.terminate()
        frontend_process.terminate()
        logger.info("All servers stopped successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch runner control panel for Smart Email Spam Detection System."
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Run the model training pipeline.")
    group.add_argument("--backend", action="store_true", help="Launch the FastAPI backend server.")
    group.add_argument("--frontend", action="store_true", help="Launch the Streamlit dashboard frontend.")
    group.add_argument("--all", action="store_true", help="Launch both servers concurrently.")
    
    args = parser.parse_args()
    
    if args.train:
        run_training()
    elif args.backend:
        run_backend()
    elif args.frontend:
        run_frontend()
    elif args.all:
        run_all()
