import httpx
import os
from .base import BaseLLMConnector

class QwenConnector(BaseLLMConnector):
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = os.getenv("QWEN_MODEL", "qwen/qwen3-coder:free")
        self.url = "https://openrouter.ai/api/v1/chat/completions"
    
    async def chat(self, messages: list[dict]) -> str:
        # Make sure messages is a list of dicts: [{"role":"user","content":"..."}, ...]
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": messages
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(self.url, json=payload, headers=headers)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                print("Error from Qwen API:", resp.text)
                raise e

            data = resp.json()
            # OpenRouter Chat API returns messages in data["choices"][0]["message"]["content"]
            return data["choices"][0]["message"]["content"]
