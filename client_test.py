"""Simple example client that posts a prompt to the local proxy server.

Usage (Windows cmd):
  python example_client.py "Hello, how are you?"

Make sure the server is running (see README) and OPENAI_API_KEY is set.
"""
import sys
import json
import requests

def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Say hello"
    payload = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
    }

    resp = requests.post("http://127.0.0.1:8000/llm/chat", json=payload, timeout=60)
    try:
        resp.raise_for_status()
    except Exception:
        print("Error from server:", resp.status_code, resp.text)
        return

    data = resp.json()
    # Print the full response data if you want to see all fields
    print("\nFull response:")
    print(json.dumps(data, indent=2))
    
    # Print just the response text, which will properly render escape sequences
    print("\nAssistant response:")
    print(data["response"])


if __name__ == "__main__":
    main()
