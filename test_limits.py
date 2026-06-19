import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv("backend/.env")
api_key = os.getenv("GEMINI_API_KEY", "").strip(' "\'')
client = genai.Client(api_key=api_key)

models_to_test = ['gemini-2.5-flash', 'gemini-flash-latest', 'gemini-2.0-flash']

for model in models_to_test:
    print(f"Testing {model}...")
    try:
        response = client.models.generate_content(
            model=model,
            contents='Say hello',
            config=types.GenerateContentConfig(temperature=0.1)
        )
        print(f"Success for {model}: {response.text}")
    except Exception as e:
        print(f"Error for {model}: {e}")
