import httpx
import os
from .base import BaseLLMConnector

class MistralConnector(BaseLLMConnector):
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = os.getenv("MISTRAL_MODEL", "mistralai/mistral-small-3.2-24b-instruct:free")  # default free model
        self.url = "https://openrouter.ai/api/v1/chat/completions"
    
    async def chat(self, messages: list[dict]) -> str:
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
                print("Error from OpenRouter API:", resp.text)
                raise e

            data = resp.json()
            # OpenRouter returns messages under choices[0].message.content
            return data["choices"][0]["message"]["content"]
