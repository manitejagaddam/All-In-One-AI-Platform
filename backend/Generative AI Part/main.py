from fastapi import FastAPI, HTTPException
import asyncio
import logging
import os
from dotenv import load_dotenv

from models import Message, ChatRequest, MultiChatRequest
from llm_connectors.deepseek import DeepSeekConnector
from llm_connectors.mistral import MistralConnector
from llm_connectors.qwen import QwenConnector
from utils import generate_session_id

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI
app = FastAPI(title="Multi-LLM Chat API")

# Initialize connectors and validate API keys
connectors = {}
for name, cls in [("deepseek", DeepSeekConnector), 
                  ("mistral", MistralConnector), 
                  ("qwen", QwenConnector)]:
    instance = cls()
    if not instance.api_key:
        raise RuntimeError(f"API key for {name} not found! Check your .env")
    connectors[name] = instance

# ------------------- Endpoints -------------------

@app.post("/chat")
async def chat(req: ChatRequest):
    connector = connectors.get(req.model)
    if not connector:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {req.model}")

    session_id = req.session_id or generate_session_id()
    messages_dicts = [m.dict() for m in req.messages]

    try:
        reply = await connector.chat(messages_dicts, session_id=session_id)
        logging.info(f"Reply from {req.model}: {reply}")
        return {"session_id": session_id, "reply": reply}
    except Exception as e:
        logging.error(f"Error from {req.model}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chatAll")
async def multi_chat(req: MultiChatRequest):
    session_id = req.session_id or generate_session_id()
    messages_dicts = [m.dict() for m in req.messages]

    async def call_model(model_name: str):
        connector = connectors.get(model_name)
        if not connector:
            return model_name, {"error": "Unsupported model"}
        try:
            reply = await connector.chat(messages_dicts, session_id=session_id)
            return model_name, {"reply": reply}
        except Exception as e:
            logging.error(f"Error from {model_name}: {e}")
            return model_name, {"error": str(e)}

    results = await asyncio.gather(*(call_model(m) for m in req.models))
    return {"session_id": session_id, "responses": dict(results)}
