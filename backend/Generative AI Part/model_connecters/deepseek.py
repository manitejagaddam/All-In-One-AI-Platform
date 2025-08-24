import requests


class DeepSeekConnection:
    def __init__(self, api_key:str):
        self.api = api_key
        self.url = "https://api.deepseek.com/v1/chat/completions"
        
    def send(self, messages, model = "deepseek-chat"):
        headers = {"Authorization": f"Bearer {self.api}"}
        payload = {
            "model": model,
            "messages": messages,
        }
        r = requests.post(self.url, json=payload, headers=headers, timeout=10)
        r.raise_for_status()
        
        j = r.json()
        
        return j["choices"][0]["message"]["context"]