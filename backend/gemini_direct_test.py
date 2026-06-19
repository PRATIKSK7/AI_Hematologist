import os
import sys
from dotenv import load_dotenv

# Load .env
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GEMINI_API_KEY")

print(f"API Key Found: {bool(api_key)}")
if api_key:
    print(f"API Key Length: {len(api_key)}")
    print(f"API Key starts with: {api_key[:5]}...")
    if api_key.strip() != api_key:
        print("WARNING: API Key has leading or trailing whitespace!")
    if "\n" in api_key or "\r" in api_key:
        print("WARNING: API Key contains hidden newline characters!")

try:
    from google import genai
    import httpx
    
    # We create an interceptor to dump the network trace
    def request_logger(request):
        print(f"\n--- NETWORK REQUEST ---")
        print(f"URL: {request.url}")
        print(f"Method: {request.method}")
        print(f"Headers: {request.headers}")
        print(f"-----------------------\n")
        
    client = genai.Client(
        api_key=api_key,
        http_options={'httpx_client': httpx.Client(event_hooks={'request': [request_logger]})}
    )
    
    print("Executing generate_content...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Say hello"
    )
    print("Response received successfully!")
    print(response)
except Exception as e:
    print("\n--- ERROR CAUGHT ---")
    import traceback
    traceback.print_exc()
