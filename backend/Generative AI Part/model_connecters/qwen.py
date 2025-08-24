from qwen_api.client import Qwen
from qwen_api.types.chat import ChatMessage

class qwenConnection:
    def __init__(self, api_key : str):
        self.client = Qwen(api_key=api_key)
    
    def send(self, messages, model="qwen2.5-omni-7b"):
        chat_msgs = [ChatMessage(role=m["role"], content=m["content"]) for m in messages]
        resp = self.client.chat.create(messages=chat_msgs, model=model)
        return resp.choices[0].message.content
        