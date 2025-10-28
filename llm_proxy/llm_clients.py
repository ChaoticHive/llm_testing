"""LLM provider adapters and dispatcher.

Each adapter follows the same interface:
- Input: (api_key, model, messages, temperature)
- Output: assistant response text (str)
- Raises: on API/auth errors

Providers supported:
- OpenAI (ChatGPT)
- Google Gemini
- Anthropic Claude
- Azure OpenAI
"""
import os
import json
from typing import List, Dict, Any, Optional
import httpx

# Provider-specific imports (install via requirements.txt)
try:
    from google.oauth2 import service_account
    from google.auth.transport import requests as google_requests
    HAVE_GOOGLE = True
except ImportError:
    HAVE_GOOGLE = False

try:
    import anthropic
    HAVE_ANTHROPIC = True
except ImportError:
    HAVE_ANTHROPIC = False

try:
    from azure.identity import DefaultAzureCredential
    HAVE_AZURE = True
except ImportError:
    HAVE_AZURE = False


# Configuration and endpoints
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
GEMINI_API_URL = os.getenv("GEMINI_API_URL", "https://generativelanguage.googleapis.com/v1/models")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")


def call_openai(api_key: str, model: str, messages: List[Dict[str, Any]], temperature: float = 0.0) -> str:
    """Call OpenAI Chat Completions API."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(OPENAI_CHAT_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("No choices in OpenAI response")

    message = choices[0].get("message") or {}
    return message.get("content", "")


def call_gemini(api_key: Optional[str], model: str, messages: List[Dict[str, Any]], temperature: float = 0.0) -> str:
    """Call Google's Gemini API using either API key or service account."""
    if not HAVE_GOOGLE:
        raise RuntimeError("Gemini support requires: pip install google-auth")

    # Format messages into Gemini's expected shape
    prompt = ""
    for msg in messages:
        role, content = msg["role"], msg["content"]
        if role == "system":
            prompt += f"Instructions: {content}\n\n"
        elif role == "user":
            prompt += f"User: {content}\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n"
    
    # Support both API key and service account auth
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        # Try service account from env
        creds = service_account.Credentials.from_service_account_file(
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        auth_req = google_requests.Request()
        creds.refresh(auth_req)
        headers["Authorization"] = f"Bearer {creds.token}"

    payload = {
        "contents": [{
            "parts":[{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": temperature,
        },
    }

    url = f"{GEMINI_API_URL}/{model}:generateContent"
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected Gemini response format: {e}")


def call_anthropic(api_key: str, model: str, messages: List[Dict[str, Any]], temperature: float = 0.0) -> str:
    """Call Anthropic's Claude API."""
    if not HAVE_ANTHROPIC:
        raise RuntimeError("Anthropic support requires: pip install anthropic")

    client = anthropic.Client(api_key=api_key)
    
    # Convert our messages to Anthropic's format
    system = next((m["content"] for m in messages if m["role"] == "system"), None)
    history = [m for m in messages if m["role"] != "system"]
    
    response = client.messages.create(
        model=model,
        max_tokens=1000,
        temperature=temperature,
        system=system,
        messages=[
            {"role": "user" if m["role"] == "user" else "assistant", "content": m["content"]}
            for m in history
        ],
    )
    return response.content[0].text


def call_azure_openai(api_key: Optional[str], model: str, messages: List[Dict[str, Any]], temperature: float = 0.0) -> str:
    """Call Azure OpenAI using either API key or Azure credential."""
    if not HAVE_AZURE:
        raise RuntimeError("Azure OpenAI support requires: pip install azure-identity")

    if not AZURE_OPENAI_ENDPOINT:
        raise ValueError("AZURE_OPENAI_ENDPOINT environment variable not set")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["api-key"] = api_key
    else:
        # Use Azure identity (managed identity or service principal)
        credential = DefaultAzureCredential()
        token = credential.get_token("https://cognitiveservices.azure.com/.default")
        headers["Authorization"] = f"Bearer {token.token}"

    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{model}/chat/completions?api-version=2023-05-15"
    payload = {
        "messages": messages,
        "temperature": temperature,
    }

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("No choices in Azure OpenAI response")

    message = choices[0].get("message") or {}
    return message.get("content", "")


def dispatch_call(provider: str, api_key: str, model: str, messages: List[Dict[str, Any]], temperature: float = 0.0) -> str:
    """Route requests to the appropriate provider adapter."""
    provider = provider.lower()
    
    # Use provider-specific API key from env if not provided
    if not api_key:
        env_key = os.getenv(f"{provider.upper()}_API_KEY")
        if not env_key:
            raise ValueError(f"No API key provided and {provider.upper()}_API_KEY not set in environment")
        api_key = env_key

    if provider in ("openai", "chatgpt"):
        return call_openai(api_key, model, messages, temperature)
    elif provider == "gemini":
        return call_gemini(api_key, model, messages, temperature)
    elif provider == "anthropic":
        return call_anthropic(api_key, model, messages, temperature)
    elif provider == "azure":
        return call_azure_openai(api_key, model, messages, temperature)
    
    raise ValueError(f"Unsupported provider: {provider}")


# For backward compatibility
call_chatgpt = call_openai  # Alias old name to new function
