import httpx
import os 
from .base import BaseLLMConnector


class DeepSeekConnector(BaseLLMConnector):
    def __init__(self, model = "deepseek-chat"):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.url = "https://api.deepseek.com/v1/chat/completions"
        self.model = model
        
    async def chat(self, messages : list[dict]) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": messages
        }
        
        async with httpx.AsyncClient() as client:
            resp = await client.post(self.url, json=payload, headers=headers)
            resp.raise_for_status()
            
            data = resp.json()
            
            return data["choices"][0]["message"]["content"]