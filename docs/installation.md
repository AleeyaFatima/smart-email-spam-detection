# Installation & Setup Guide
## Smart Email Spam Detection System

Follow these steps to set up, train, and run the Smart Email Spam Detection System on your computer.

---

### Prerequisites
1.  **Python**: A stable version of Python (preferably **Python 3.11** or **Python 3.12**).
2.  **uv Package Manager**: Astral `uv` is a super-fast Python package manager used to set up the environment in seconds.
    *   To install `uv` on Windows, run this in PowerShell:
        ```powershell
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        ```
    *   To install on macOS/Linux, run:
        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        ```

---

### Step 1: Initialize Virtual Environment
Open your terminal inside the project directory (`Smart Email Spam Detection System/`) and run the following command to create a virtual environment running Python 3.11:

```bash
uv venv --python 3.11
```

This will create a `.venv/` directory inside your project folder.

---

### Step 2: Install Package Dependencies
Install the required tools listed in `requirements.txt` into the virtual environment by running:

```bash
uv pip install -r requirements.txt
```

*This will take under 5 seconds, downloading precompiled binary wheels for scikit-learn, gensim, pandas, fastapi, and streamlit.*

---

### Step 3: Run Model Training & Comparison Pipeline
Before running the dashboard, you must train the machine learning models. This script will download the benchmark dataset, preprocess all texts, train 8 model-vectorizer combinations, and write the evaluation cards.

Run the training command:
```bash
.venv\Scripts\python.exe run.py --train
```
*(On macOS/Linux, run `source .venv/bin/activate` followed by `python run.py --train`)*

---

### Step 4: Run the Complete System Concurrently
To launch both the **FastAPI Backend REST API** and the **Streamlit Dashboard User Interface** at the same time, run:

```bash
.venv\Scripts\python.exe run.py --all
```

*The console will show that Uvicorn and Streamlit are online.*

---

### Accessing the Web Portals
Once the servers are online, you can access them at:
*   **Streamlit Interactive Dashboard**: [http://127.0.0.1:8501](http://127.0.0.1:8501)
*   **FastAPI Swagger API Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
*   **FastAPI Root Health Check**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

Press **Ctrl+C** in the terminal to stop both servers concurrently.
