"""
    -Gemini_Config-
"""

from app.core.env import settings
from google import genai

apiKey = settings.GEMINI_API_KEY
genAI = genai.Client(api_key=apiKey)