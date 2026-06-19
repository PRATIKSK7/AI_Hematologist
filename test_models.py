import os
from google import genai
from dotenv import load_dotenv

load_dotenv("backend/.env")
api_key = os.getenv("GEMINI_API_KEY", "").strip(' "\'')
client = genai.Client(api_key=api_key)

try:
    for model in client.models.list():
        print(model.name)
except Exception as e:
    print(f"Error: {e}")
