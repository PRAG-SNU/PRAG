import os

# API Key Configuration
OPENAI_API_KEY = "your_openai_api_key_here"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# Base directory (automatically set to the directory containing config.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths relative to BASE_DIR
DATA_PATH = os.path.join(BASE_DIR, "data")
DB_DIRECTORY = os.path.join(BASE_DIR, "database/photosyn_papers_db")
MODEL_NAME = "gpt-4o"  # or 'gpt-3.5-turbo', etc.