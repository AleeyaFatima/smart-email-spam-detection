import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.config import LOG_FORMAT, LOG_FILE
from src.database.db import init_db
from src.backend.routes import router

# Setup Backend Logging
# Logging to both the console and a file represents standard production practice
import os
log_handlers = [logging.StreamHandler()]
if not os.environ.get("VERCEL"):
    try:
        log_handlers.append(logging.FileHandler(LOG_FILE, encoding="utf-8"))
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=log_handlers
)
logger = logging.getLogger(__name__)


# ==========================================
# DESIGN DECISION: Lifespan Context Manager
# We use the modern FastAPI 'lifespan' pattern (which replaces the deprecated
# @app.on_event('startup') event listeners) to run database migrations
# before the server starts accepting HTTP requests.
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages startup and shutdown routines.
    Runs database initialization on startup.
    """
    logger.info("Starting up FastAPI application...")
    try:
        # Initialize the SQLite tables
        init_db()
        logger.info("Database migration check passed.")
    except Exception as e:
        logger.critical(f"Failed to initialize database during startup: {e}")
        
    yield  # Hand over control to the web server
    
    logger.info("Shutting down FastAPI application...")


app = FastAPI(
    title="Smart Email Spam Detection System API",
    description=(
        "Production-quality REST API server for real-time and batch email spam detection. "
        "Built using FastAPI, SQLite, Scikit-Learn, and Gensim. "
        "Supports dynamic re-training, prediction history logging, and metric comparison."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS (Cross-Origin Resource Sharing)
# Streamlit will run on port 8501, and FastAPI runs on port 8000.
# We must allow Streamlit's address to send requests to FastAPI.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify local domains for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router)

# Mount the static directory to serve index.html at root "/"
# This serves our premium custom HTML/CSS/JS dashboard interface.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
