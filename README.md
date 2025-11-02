# LLM Proxy (ChatGPT starter)

This small project provides a tiny FastAPI server that can proxy requests to different LLM providers. It currently implements a ChatGPT (OpenAI) client and exposes a single endpoint you can call locally.

Files
- `llm_proxy/app.py` - FastAPI app exposing POST `/llm/chat`.
- `llm_proxy/llm_clients.py` - minimal OpenAI Chat Completions HTTP client.
- `example_client.py` - small script to POST to the running server for testing.
- `requirements.txt` - Python dependencies.
- `.env.example` - example environment variable settings.

````markdown
# LLM Proxy (ChatGPT starter)

This small project provides a tiny FastAPI server that can proxy requests to different LLM providers. It currently implements a ChatGPT (OpenAI) client and exposes a single endpoint you can call locally.

Files
- `llm_proxy/app.py` - FastAPI app exposing POST `/llm/chat` and Redis-backed conversation endpoints.
- `llm_proxy/llm_clients.py` - minimal OpenAI Chat Completions HTTP client.
- `example_client.py` - small script to POST to the running server for testing.
- `example_conversation.py` - example usage for conversation endpoints (create/send/get).
- `requirements.txt` - Python dependencies.
- `.env.example` - example environment variable settings.

Quick start (Windows cmd)

1. Create virtual env and activate

```
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Set your OpenAI API key (replace with your key)

```
set OPENAI_API_KEY=sk-...your-key...
```

4. Run the server

```
python -m uvicorn llm_proxy.app:app --reload --port 8000
```

5. In another cmd window run the example client

```
python example_client.py "Hello from local proxy"
```

API

POST /llm/chat
- JSON body:
  - provider: "openai" or "chatgpt" (case-insensitive)
  - model: model id (e.g. "gpt-4o-mini")
  - messages: list of chat messages (OpenAI chat format: each item has role and content)
  - temperature: optional float

Response: JSON with fields `provider`, `model`, and `response` (assistant string).

Extending

To add another provider, implement a client function with the same signature as `call_chatgpt` in `llm_proxy/llm_clients.py` and wire it in `app.py` under a new provider name.

Redis-backed conversations (new)

This project now includes Redis-backed conversation storage and endpoints. Use these endpoints when you want the server to persist multi-turn conversations between requests.

Environment variables
- `REDIS_URL` - Redis connection URL (default: `redis://localhost:6379/0`)
- `MAX_HISTORY_LENGTH` - maximum number of messages to keep for a conversation (default: 200)

Endpoints
- POST /conversations
  - Body: { "instructions": <optional system prompt>, "messages": <optional initial messages array>, "provider": <optional>, "model": <optional>, "temperature": <optional> }
  - Returns: { "conversation_id": "...", "messages": [...] }

- GET /conversations/{conversation_id}
  - Returns conversation messages

- POST /conversations/{conversation_id}/messages
  - Body: { "role": "user"|"assistant"|..., "content": "..." }
  - Optional query/body overrides: provider, model, temperature
  - Behavior: appends the incoming message, calls the configured LLM provider, appends the assistant response, saves history, and returns the assistant reply and updated messages

- DELETE /conversations/{conversation_id}
  - Deletes the conversation and metadata

Quick demo (Windows cmd)

1) Start server (ensure Redis is running locally or set REDIS_URL):

```
set OPENAI_API_KEY=sk-...
set REDIS_URL=redis://localhost:6379/0
python -m uvicorn llm_proxy.app:app --reload --port 8000
```

2) Create a conversation and send messages (in another window):

```
python example_conversation.py create "You are concise and friendly."
# copy the returned conversation_id
python example_conversation.py send <conversation_id> "Hello!"
python example_conversation.py get <conversation_id>

Live push-to-talk (tap-to-toggle)

This repository includes an example script that captures microphone audio, transcribes it with OpenAI Whisper, and sends the transcript to the proxy while maintaining a conversation on the server.

By default the script reuses the same conversation id for the life of the program and persists the id to `.llm_conversation.json` in the current working directory. You can change that path with `--persist-file` and resume a saved conversation with `--resume`.

Run the proxy (make sure the proxy terminal has your OPENAI_API_KEY set for completions):

```cmd
cd "f:\Backend LLM"
set OPENAI_API_KEY=sk-...your-key...
python -m uvicorn llm_proxy.app:app --reload --port 8000
```

Run the push-to-talk script (transcription also needs OPENAI_API_KEY):

```cmd
cd "f:\Backend LLM"
set OPENAI_API_KEY=sk-...your-key...
python examples\live_push_to_talk.py --key F8 --provider openai --model gpt-4 --instructions "You are concise."
```

To persist the conversation id and resume it later:

```cmd
python examples\live_push_to_talk.py --persist-file my_conv.json --resume
```

Example session

1) Start the proxy and the push-to-talk script (in separate terminals) as shown above.

2) Tap the hotkey once to start recording, speak a short question, then tap again to stop. You should see something like:

```
Recording started (tap F8 again to stop)
Saved audio to C:\Users\you\AppData\Local\Temp\tmpabcd1234.wav; transcribing...
--- Transcript ---
What's the weather like in Paris?
------------------
Sending transcript to proxy...
--- Assistant Response ---
In Paris today it's sunny with a high of 22°C. Would you like a 3-day forecast?
---------------------------

=== Current Conversation ===
[System] You are concise.
[You] What's the weather like in Paris?
[Assistant] In Paris today it's sunny with a high of 22°C. Would you like a 3-day forecast?
===========================
```

3) Later you can resume the same conversation in a new run of the script with:

```cmd
python examples\live_push_to_talk.py --persist-file my_conv.json --resume
```

4) Or use `example_conversation.py` or any client to send messages to that conversation id (the file `my_conv.json` contains the id).


```

Notes
- This uses Redis to persist conversation state. For production, secure access to Redis, add authentication, and consider persistence/backup strategies.
- The implementation does a simple truncation to `MAX_HISTORY_LENGTH`. For long-running conversations consider summarization or selective context strategies.

````
