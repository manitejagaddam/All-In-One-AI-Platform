from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import asyncio
import logging

from llm_connectors.deepseek import DeepSeekConnector
from llm_connectors.mistral import MistralConnector
from llm_connectors.qwen import QwenConnector

# Configure logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Multi-LLM Chat API")

# Initialize connectors
connectors = {
    "deepseek": DeepSeekConnector(),
    "mistral": MistralConnector(),
    "qwen": QwenConnector()
}

# Pydantic models
class Message(BaseModel):
    role: str = Field(..., description="Role of the message: 'user' or 'assistant'")
    content: str = Field(..., description="Text content of the message")

class ChatRequest(BaseModel):
    model: str = Field(..., description="The LLM model to use")
    messages: List[Message]
    session_id: Optional[str] = Field(None, description="Optional session ID for context")

class MultiChatRequest(BaseModel):
    models: List[str]
    messages: List[Message]
    session_id: Optional[str] = None

# Single LLM chat
@app.post("/chat")
async def chat(req: ChatRequest):
    connector = connectors.get(req.model)
    if not connector:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {req.model}")

    try:
        reply = await connector.chat(req.messages, session_id=req.session_id)
        logging.info(f"Reply from {req.model}: {reply}")
        return {"reply": reply}
    except Exception as e:
        logging.error(f"Error from {req.model}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Multi-LLM chat (parallel)
@app.post("/chatAll")
async def multi_chat(req: MultiChatRequest):
    async def call_model(model_name):
        connector = connectors.get(model_name)
        if not connector:
            return model_name, {"error": "Unsupported model"}
        try:
            reply = await connector.chat(req.messages, session_id=req.session_id)
            return model_name, {"reply": reply}
        except Exception as e:
            logging.error(f"Error from {model_name}: {e}")
            return model_name, {"error": str(e)}

    results = await asyncio.gather(*(call_model(m) for m in req.models))
    return dict(results)
