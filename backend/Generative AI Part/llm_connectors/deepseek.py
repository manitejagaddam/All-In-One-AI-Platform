import httpx
import os 
from dotenv import load_dotenv
from .base import BaseLLMConnector

load_dotenv()

class DeepSeekConnector(BaseLLMConnector):
    def __init__(self):
        # Load key from env, fallback to hardcoded for testing
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        print("API Key loaded:", self.api_key[:10] + "...")
        
        # Correct endpoint
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        print("Endpoint:", self.url)
        
        # Default model
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek/deepseek-r1:free")
        print("Using model:", self.model)
        
    async def chat(self, messages: list[dict]) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(self.url, json=payload, headers=headers)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                print("Error from DeepSeek API:", resp.text)
                raise e
            
            data = resp.json()
            print("DeepSeek Response:", data)
            
            return data["choices"][0]["message"]["content"]
