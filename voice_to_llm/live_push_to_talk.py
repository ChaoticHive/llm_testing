"""Push-to-talk live microphone -> transcription -> LLM proxy example.

Usage (Windows cmd):
    set OPENAI_API_KEY=sk-...
    python examples/live_push_to_talk.py --key F8 --provider openai --model gpt-4 --instructions "Be concise." --server-url http://localhost:8000

Controls:
 - Tap the configured key (default F8) once to start recording, tap again to stop and send.
 - Press ESC to exit the program.

Notes:
 - This records PCM 16-bit audio using `sounddevice` and writes a temporary WAV file for transcription.
 - Transcription uses OpenAI's transcription endpoint (OPENAI_API_KEY required).
 - The script reuses the same conversation id for the whole program run. By default it persists the id
     to `.llm_conversation.json` in the current directory; pass `--persist-file` to change location or
     `--resume` to load an existing file on startup.
 - If you prefer other transcription services, replace `transcribe_with_openai`.
"""

import argparse
import os
import sys
import tempfile
import threading
import time
import wave

import httpx
import json
from typing import Optional
import keyboard
import numpy as np
import sounddevice as sd

OPENAI_TRANSCRIBE_URL = "https://api.openai.com/v1/audio/transcriptions"


class ProxyConversation:
    def __init__(self, server_url: str, provider: str, model: str, instructions: Optional[str] = None, temperature: float = 0.0):
        self.server_url = server_url.rstrip('/')
        self.provider = provider
        self.model = model
        self.instructions = instructions
        self.temperature = temperature
        self.conversation_id = None
        self._messages = []

    def create(self) -> str:
        """Create a new conversation on the proxy server."""
        url = f"{self.server_url}/conversations"
        payload = {
            "provider": self.provider,
            "model": self.model,
            "instructions": self.instructions,
            "temperature": self.temperature
        }
        
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            self.conversation_id = data["conversation_id"]
            self._messages = data["messages"]
            return self.conversation_id

    def load(self, conversation_id: str) -> str:
        """Load an existing conversation by id and fetch its messages from the proxy."""
        url = f"{self.server_url}/conversations/{conversation_id}"
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()
            # The GET returns {'conversation_id', 'messages'}
            self.conversation_id = conversation_id
            self._messages = data.get("messages", [])
            return self.conversation_id

    def save_to_file(self, path: str):
        """Persist conversation id and metadata to a small JSON file."""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({
                    'conversation_id': self.conversation_id,
                    'provider': self.provider,
                    'model': self.model,
                }, f)
        except Exception as e:
            print(f"Warning: failed to save conversation to {path}: {e}")

    def send_message(self, text: str) -> str:
        """Send a message in the conversation and return the assistant's response."""
        if not self.conversation_id:
            self.create()

        url = f"{self.server_url}/conversations/{self.conversation_id}/messages"
        payload = {"role": "user", "content": text}
        
        print(f"\nDebug: Sending to {url}")
        print(f"Debug: Message = {text}")
        print(f"Debug: Conversation ID = {self.conversation_id}")
        
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                self._messages = data["messages"]
                return data["assistant_response"]
        except httpx.HTTPStatusError as e:
            print(f"\nProxy server error: {e.response.status_code} {e.response.reason_phrase}")
            print(f"Response body: {e.response.text}")
            if "OPENAI_API_KEY" in str(e.response.text):
                print("\nLikely cause: OPENAI_API_KEY environment variable is not set on the proxy server.")
            raise
        except httpx.ConnectError:
            print(f"\nCould not connect to proxy at {self.server_url}")
            print("Make sure the proxy server is running:")
            print("  cd Backend LLM")
            print("  python -m uvicorn llm_proxy.app:app --reload --port 8000")
            raise

    @property
    def messages(self) -> list:
        """Get all messages in the conversation."""
        return self._messages.copy() if self._messages else []


class Recorder:
    def __init__(self, samplerate=16000, channels=1, dtype='int16'):
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self._frames = []
        self._stream = None
        self._lock = threading.Lock()

    def _callback(self, indata, frames, time_info, status):
        # indata is a numpy array with shape (frames, channels)
        if status:
            print(f"[recorder] status: {status}")
        # Copy the buffer to avoid later mutation
        with self._lock:
            self._frames.append(indata.copy())

    def start(self):
        self._frames = []
        self._stream = sd.InputStream(samplerate=self.samplerate, channels=self.channels, dtype=self.dtype,
                                      callback=self._callback)
        self._stream.start()

    def stop(self):
        if self._stream is None:
            return None
        self._stream.stop()
        self._stream.close()
        self._stream = None
        # Concatenate frames into a single numpy array
        with self._lock:
            if not self._frames:
                return np.empty((0, self.channels), dtype=self.dtype)
            audio = np.concatenate(self._frames, axis=0)
            self._frames = []
        return audio


def write_wav(path: str, audio: np.ndarray, samplerate: int, channels: int, sampwidth=2):
    """Write numpy int16 array to a WAV file."""
    # Ensure int16
    if audio.dtype != np.int16:
        # If float, convert
        if np.issubdtype(audio.dtype, np.floating):
            audio_out = (audio * 32767.0).astype(np.int16)
        else:
            audio_out = audio.astype(np.int16)
    else:
        audio_out = audio

    with wave.open(path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(samplerate)
        wf.writeframes(audio_out.tobytes())


def transcribe_with_openai(audio_path: str, openai_api_key: str, model: str = "whisper-1") -> str:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    headers = {"Authorization": f"Bearer {openai_api_key}"}
    try:
        with open(audio_path, 'rb') as f:
            files = {"file": (os.path.basename(audio_path), f, "application/octet-stream")}
            data = {"model": model}
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(OPENAI_TRANSCRIBE_URL, headers=headers, data=data, files=files)
        resp.raise_for_status()
        return resp.json().get("text", "")
    except httpx.HTTPStatusError as e:
        print(f"OpenAI transcription error: {e.response.status_code} {e.response.reason_phrase}")
        print(f"Response body: {e.response.text}")
        if e.response.status_code == 401:
            print("Likely cause: Invalid or missing OPENAI_API_KEY.")
        elif e.response.status_code == 400:
            print("Likely cause: Bad request. Check audio format (must be WAV/MP3/M4A/OGG), file size, or model name.")
        elif e.response.status_code == 429:
            print("Likely cause: Quota exceeded or rate limited.")
        elif e.response.status_code == 500:
            print("Likely cause: OpenAI server error. Try again later or check file format/model.")
        raise
    except Exception as e:
        print(f"Unexpected error during transcription: {e}")
        raise



def main():
    parser = argparse.ArgumentParser(description="Push-to-talk live microphone -> transcription -> LLM proxy")
    parser.add_argument('--key', default='F8', help='Hotkey to hold for push-to-talk (default: F8)')
    parser.add_argument('--provider', default='openai', help='Provider to send to proxy (default: openai)')
    parser.add_argument('--model', default='gpt-4', help='Model to send to proxy (default: gpt-4)')
    parser.add_argument('--instructions', default=None, help='Optional system instructions')
    parser.add_argument('--server-url', default='http://localhost:8000', help='LLM proxy URL (default http://localhost:8000)')
    parser.add_argument('--samplerate', type=int, default=16000, help='Audio sample rate (default: 16000)')
    parser.add_argument('--channels', type=int, default=1, help='Audio channels (default: 1)')
    parser.add_argument('--openai-model', default='whisper-1', help='OpenAI transcription model (default: whisper-1)')
    parser.add_argument('--new-conversation', action='store_true', help='Start a new conversation (default: continue last one)')
    parser.add_argument('--persist-file', default='.llm_conversation.json', help='Path to JSON file to save/load conversation id (default: .llm_conversation.json)')
    parser.add_argument('--resume', action='store_true', help='If set and persist-file exists, load that conversation id instead of creating a new one')

    args = parser.parse_args()
    
    args.instructions = """
    You are a recovering gambling addict. Trying to quit gambling is hard, but you are determined to turn your life around.
    You understand the triggers and challenges associated with gambling addiction, and you are committed to finding
    healthier coping mechanisms and support systems to maintain your sobriety.
    """
    
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print('ERROR: OPENAI_API_KEY must be set for transcription')
        sys.exit(1)

    hotkey = args.key
    recorder = Recorder(samplerate=args.samplerate, channels=args.channels, dtype='int16')
    
    # Create or load conversation
    conversation = ProxyConversation(
        server_url=args.server_url,
        provider=args.provider,
        model=args.model,
        instructions=args.instructions,
        temperature=0.0
    )
    
    # Handle persistence/resume logic
    if args.persist_file and args.resume and os.path.exists(args.persist_file):
        try:
            with open(args.persist_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                conv_id = data.get('conversation_id')
            if conv_id:
                conversation.load(conv_id)
                print(f"\nResumed conversation: {conversation.conversation_id}")
            else:
                # fallback to creating new
                conversation.create()
                print(f"\nCreated new conversation: {conversation.conversation_id}")
                if args.persist_file:
                    conversation.save_to_file(args.persist_file)
        except Exception as e:
            print(f"Error resuming conversation from {args.persist_file}: {e}")
            print("Creating a new conversation instead...")
            try:
                conversation.create()
                print(f"\nCreated new conversation: {conversation.conversation_id}")
                if args.persist_file:
                    conversation.save_to_file(args.persist_file)
            except Exception as e2:
                print(f"Error creating conversation: {e2}")
                sys.exit(1)
    else:
        # Create a new conversation and optionally persist its id
        try:
            conversation.create()
            print(f"\nCreated new conversation: {conversation.conversation_id}")
            if args.persist_file:
                conversation.save_to_file(args.persist_file)
        except Exception as e:
            print(f"Error creating conversation: {e}")
            sys.exit(1)

    is_recording = False
    stop_event = threading.Event()
    # Debounce toggles to avoid rapid double-trigger when holding the key
    last_toggle_time = 0.0
    TOGGLE_DEBOUNCE = 0.25  # seconds

    def start_recording_thread():
        """Start the background recording/transcription/send thread."""
        def _record_thread():
            nonlocal is_recording
            tmp_path = None
            try:
                print("Recording...")
                recorder.start()
                # Wait until stop_event is set
                stop_event.wait()
                audio = recorder.stop()
                if audio is None or audio.size == 0:
                    print("No audio captured")
                    return

                # Write temp wav
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tf:
                    tmp_path = tf.name
                write_wav(tmp_path, audio, args.samplerate, args.channels)
                print(f"Saved audio to {tmp_path}; transcribing...")

                try:
                    transcript = transcribe_with_openai(tmp_path, openai_key, model=args.openai_model)
                except Exception as ex:
                    print(f"Error during transcription: {ex}")
                    return
                print('--- Transcript ---')
                print(transcript)
                print('------------------')

                # Send to proxy in conversation
                try:
                    print('Sending transcript to proxy...')
                    print(f'Conversation ID: {conversation.conversation_id}')
                    response = conversation.send_message(transcript)
                    print('--- Assistant Response ---')
                    print(response)
                    print('---------------------------')

                    # Show conversation history
                    print('\n=== Current Conversation ===')
                    for msg in conversation.messages:
                        role = msg["role"]
                        content = msg["content"]
                        if role == "system":
                            print(f"[System] {content}")
                        elif role == "user":
                            print(f"[You] {content}")
                        elif role == "assistant":
                            print(f"[Assistant] {content}")
                    print('===========================\n')
                except Exception as ex:
                    print(f"Error sending transcript to proxy: {ex}")
            finally:
                # cleanup temp file
                try:
                    if tmp_path and os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
                is_recording = False

        t = threading.Thread(target=_record_thread, daemon=True)
        t.start()

    def toggle_recording(e):
        """Toggle recording on/off when the hotkey is pressed (tap-to-toggle)."""
        nonlocal is_recording, stop_event
        nonlocal last_toggle_time
        now = time.time()
        # simple debounce to avoid repeated events
        if now - last_toggle_time < TOGGLE_DEBOUNCE:
            return
        last_toggle_time = now

        if not is_recording:
            is_recording = True
            stop_event.clear()
            start_recording_thread()
            print(f"Recording started (tap {hotkey} again to stop)")
        else:
            stop_event.set()
            print('Recording stopped (processing...)')

    print('Push-to-talk live example')
    print(f'Tap {hotkey} to start/stop recording; press ESC to exit.')
    print(f'Conversation {conversation.conversation_id} will be reused for all recordings until this program exits.')
    if args.persist_file:
        print(f'Conversation id is stored in: {args.persist_file} (use --resume to load it on next run)')

    # Register hotkey handler (toggle)
    keyboard.on_press_key(hotkey, lambda e: toggle_recording(e))

    try:
        # main loop, exit on ESC
        while True:
            if keyboard.is_pressed('esc'):
                print('ESC pressed, exiting...')
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            keyboard.unhook_all()
        except Exception:
            pass


if __name__ == '__main__':
    main()
