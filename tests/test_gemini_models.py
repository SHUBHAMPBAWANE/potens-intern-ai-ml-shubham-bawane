from dotenv import load_dotenv
from google import genai
import os


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

print("=" * 70)
print("Available Gemini Models")
print("=" * 70)

for model in client.models.list():
    print(model.name)

print("=" * 70)