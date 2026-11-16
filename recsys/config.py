"""Paths and settings. Settings come from environment variables, or from Streamlit secrets on Streamlit Cloud."""

import os
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
TYPO_PATH = BASE_DIR / "models" / "typo_map.json"
EXAMPLES_PATH = BASE_DIR / "data" / "examples.json"

MAX_CHARS = 2000


def on_streamlit_cloud() -> bool:
    """True when running on Streamlit Community Cloud (the app is mounted under /mount/src there)."""
    return Path("/mount/src").exists()


def setting(name: str) -> str:
    """Value of an environment variable or Streamlit secret; empty string if not set."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        import streamlit as st

        return str(st.secrets[name])
    except Exception:
        return ""


def data_dir() -> Path:
    """Folder for the correction database. Temporary storage on Streamlit Cloud, ./data_store elsewhere."""
    if os.environ.get("APP_DATA_DIR"):
        return Path(os.environ["APP_DATA_DIR"])
    if on_streamlit_cloud():
        return Path(tempfile.gettempdir()) / "repair_app_data"
    return BASE_DIR / "data_store"
