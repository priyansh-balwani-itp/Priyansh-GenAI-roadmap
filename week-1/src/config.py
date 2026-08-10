import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
AIRFORCE_API_KEY = os.getenv("AIRFORCE_API_KEY")
AIRFORCE_BASE_URL = os.getenv("AIRFORCE_BASE_URL", "https://api.airforce/v1")
AIRFORCE_MODEL = os.getenv("AIRFORCE_MODEL", "deepseek-v3")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
