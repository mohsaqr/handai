"""
Handai Configuration
App-wide constants and defaults
"""

import os
import platform

# App Info
APP_NAME = "Handai"
APP_VERSION = "4.0"
APP_DESCRIPTION = "AI Data Transformer & Generator"

# Database + config storage (writable per-user location)
def _get_user_data_dir(app_name: str) -> str:
    system = platform.system().lower()
    home = os.path.expanduser("~")

    if system == "darwin":
        base = os.path.join(home, "Library", "Application Support")
    elif system == "windows":
        base = os.environ.get("APPDATA", os.path.join(home, "AppData", "Roaming"))
    else:
        base = os.environ.get("XDG_DATA_HOME", os.path.join(home, ".local", "share"))

    path = os.path.join(base, app_name)
    os.makedirs(path, exist_ok=True)
    return path


_DATA_DIR = os.environ.get("HANDAI_DATA_DIR") or _get_user_data_dir("Handai")

DB_FILE = os.path.join(_DATA_DIR, "handai_data.db")
CONFIG_FILE = os.path.join(_DATA_DIR, "handai_config.json")

# Default Settings
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 2048
DEFAULT_MAX_CONCURRENCY = 5
DEFAULT_TEST_BATCH_SIZE = 10
DEFAULT_MAX_RETRIES = 3

# HTTP Client Settings
HTTP_MAX_CONNECTIONS = 100
HTTP_MAX_KEEPALIVE = 20
HTTP_KEEPALIVE_EXPIRY = 30.0
HTTP_TIMEOUT = 120.0
HTTP_CONNECT_TIMEOUT = 10.0

# UI Settings
PROGRESS_LOG_LINES = 10
AUTOSAVE_FREQUENCY = 5
AUTOSAVE_FREQUENCY_BATCH = 20

# Sample Test Data
SAMPLE_DATA_COLUMNS = {
    "text": [
        "I absolutely love this product! Best purchase ever.",
        "Terrible experience. Would not recommend to anyone.",
        "It's okay, nothing special but does the job.",
        "Amazing quality and fast shipping. Very happy!",
        "Broke after one week. Complete waste of money.",
        "Decent value for the price. Could be better.",
        "This exceeded all my expectations! Fantastic!",
        "Not worth it. Save your money.",
        "Pretty good overall. Minor issues but satisfied.",
        "Outstanding service and product. Will buy again!"
    ],
    "category": ["Electronics", "Clothing", "Home", "Electronics", "Toys",
                 "Home", "Beauty", "Electronics", "Clothing", "Food"],
    "price": [299.99, 45.00, 89.50, 199.99, 24.99,
              67.00, 35.50, 599.00, 55.00, 28.99]
}

# Variation temperatures for generation
VARIATION_TEMPS = {
    "Low": 0.3,
    "Medium": 0.7,
    "High": 1.0,
    "Maximum": 1.5
}

# Type mappings for schema builder
TYPE_MAP = {
    "text": "str",
    "number": "int",
    "decimal": "float",
    "boolean": "bool",
    "date": "str",
    "list": "list",
    "json": "json"
}
