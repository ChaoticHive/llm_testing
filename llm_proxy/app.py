import os
import json
from uuid import uuid4
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

import redis

from .llm_clients import dispatch_call


class ChatRequest(BaseModel):
    provider: str
    model: str = "gpt-4o-mini"
    # messages in OpenAI chat format: [{"role": "user", "content": "..."}, ...]
    messages: List[Dict[str, Any]]
    # Optional system-level instructions to be prepended to the messages list
    instructions: Optional[str] = "You are a recovering gambling addict. Trying to quit gambling is hard, but you are determined to turn your life around. You understand the triggers and challenges associated with gambling addiction, and you are committed to finding healthier coping mechanisms and support systems to maintain your sobriety."
    temperature: float = 0.0


app = FastAPI(title="LLM Proxy", version="0.2")


@app.post("/llm/chat")
def llm_chat(req: ChatRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY environment variable is not set")

    # Build messages: if instructions provided in the request use them, else fall back to
    # DEFAULT_SYSTEM_PROMPT env var if set. The system instruction is prepended as a
    # message with role "system" so the model treats it as high-level instructions.
    messages = list(req.messages)
    system_prompt = req.instructions or os.getenv("DEFAULT_SYSTEM_PROMPT")
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    try:
        content = dispatch_call(req.provider, api_key, req.model, messages, req.temperature)
        return {"provider": req.provider, "model": req.model, "response": content}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Redis-backed conversation endpoints ---


class ConversationCreate(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    instructions: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    temperature: float = 0.0


class MessageIn(BaseModel):
    role: str
    content: str


# Redis connection (synchronous). Configure via REDIS_URL env var, default localhost.
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
# Max messages kept in history (simple truncation)
MAX_HISTORY_LENGTH = int(os.getenv("MAX_HISTORY_LENGTH", "200"))


def _conv_key(conv_id: str) -> str:
    return f"conv:{conv_id}"


def _meta_key(conv_id: str) -> str:
    return f"convmeta:{conv_id}"


@app.post("/conversations")
def create_conversation(payload: ConversationCreate):
    """Create a new conversation stored in Redis. Returns conversation_id and stored messages."""
    conv_id = uuid4().hex
    messages = payload.messages or []
    if payload.instructions:
        messages.insert(0, {"role": "system", "content": payload.instructions})

    # Save messages and metadata
    redis_client.set(_conv_key(conv_id), json.dumps(messages))
    meta = {"provider": payload.provider, "model": payload.model, "temperature": payload.temperature}
    redis_client.set(_meta_key(conv_id), json.dumps(meta))
    return {"conversation_id": conv_id, "messages": messages}


@app.get("/conversations/{conv_id}")
def get_conversation(conv_id: str):
    data = redis_client.get(_conv_key(conv_id))
    if data is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = json.loads(data)
    return {"conversation_id": conv_id, "messages": messages}


@app.post("/conversations/{conv_id}/messages")
def append_and_respond(conv_id: str, message: MessageIn, provider: Optional[str] = None, model: Optional[str] = None, temperature: Optional[float] = None):
    """Append a message (typically role=user) to the conversation, call the LLM, append assistant reply, and return it.

    Optional query/body overrides allow specifying provider/model/temperature for this single turn.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY environment variable is not set")

    key = _conv_key(conv_id)
    raw = redis_client.get(key)
    if raw is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = json.loads(raw)
    # Append incoming message
    messages.append({"role": message.role, "content": message.content})

    # Load meta and determine provider/model/temperature
    meta_raw = redis_client.get(_meta_key(conv_id))
    meta = json.loads(meta_raw) if meta_raw else {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.0}
    use_provider = (provider or meta.get("provider") or "openai").lower()
    use_model = model or meta.get("model") or "gpt-4o-mini"
    use_temp = temperature if temperature is not None else meta.get("temperature", 0.0)

    try:
        assistant_content = dispatch_call(use_provider, api_key, use_model, messages, use_temp)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Append assistant reply and trim history
    messages.append({"role": "assistant", "content": assistant_content})
    if len(messages) > MAX_HISTORY_LENGTH:
        # Keep last MAX_HISTORY_LENGTH messages
        messages = messages[-MAX_HISTORY_LENGTH:]

    redis_client.set(key, json.dumps(messages))
    return {"conversation_id": conv_id, "assistant_response": assistant_content, "messages": messages}


@app.delete("/conversations/{conv_id}")
def delete_conversation(conv_id: str):
    deleted = redis_client.delete(_conv_key(conv_id))
    redis_client.delete(_meta_key(conv_id))
    if deleted:
        return {"conversation_id": conv_id, "deleted": True}
    raise HTTPException(status_code=404, detail="Conversation not found")
