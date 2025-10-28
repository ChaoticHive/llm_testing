"""Example using Redis-backed conversation endpoints.

Usage (Windows cmd):
  1) Create a conversation:
     python example_conversation.py create "You are concise and friendly."

  2) Send a message to the conversation:
     python example_conversation.py send <conversation_id> "Hello there"

  3) Get conversation history:
     python example_conversation.py get <conversation_id>

Make sure the server is running and OPENAI_API_KEY is set.
"""
import sys
import json
import requests

BASE = "http://127.0.0.1:8000"


def create(instructions=None):
    payload = {"instructions": instructions, "messages": []}
    resp = requests.post(f"{BASE}/conversations", json=payload)
    resp.raise_for_status()
    print(json.dumps(resp.json(), indent=2))


def send(conv_id, text):
    payload = {"role": "user", "content": text}
    resp = requests.post(f"{BASE}/conversations/{conv_id}/messages", json=payload, timeout=120)
    resp.raise_for_status()
    print(json.dumps(resp.json(), indent=2))


def get(conv_id):
    resp = requests.get(f"{BASE}/conversations/{conv_id}")
    resp.raise_for_status()
    print(json.dumps(resp.json(), indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: create|send|get ...")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "create":
        instructions = sys.argv[2] if len(sys.argv) > 2 else None
        create(instructions)
    elif cmd == "send":
        if len(sys.argv) < 4:
            print("Usage: send <conversation_id> <text>")
            sys.exit(1)
        send(sys.argv[2], sys.argv[3])
    elif cmd == "get":
        if len(sys.argv) < 3:
            print("Usage: get <conversation_id>")
            sys.exit(1)
        get(sys.argv[2])
    else:
        print("Unknown command")
