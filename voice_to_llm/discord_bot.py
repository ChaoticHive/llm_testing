# discord_voice_bot.py
import os
import tempfile
import asyncio
from typing import Dict, Optional

import httpx
import discord
from discord.ext import commands

# Optional TTS backend (edge-tts). Replace with your preferred TTS if needed.
import edge_tts

PROXY_URL = os.getenv("PROXY_URL", "http://127.0.0.1:8000")
PROVIDER = os.getenv("PROVIDER", "openai")
MODEL = os.getenv("MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
DEFAULT_INSTRUCTIONS = os.getenv("INSTRUCTIONS", "You are concise and helpful.")

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Map text_channel_id -> {conv_id, instructions}
conversations: Dict[int, Dict[str, str]] = {}

async def create_conversation(instructions: Optional[str]) -> str:
    payload = {
        "provider": PROVIDER,
        "model": MODEL,
        "instructions": instructions or DEFAULT_INSTRUCTIONS,
        "temperature": TEMPERATURE,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{PROXY_URL}/conversations", json=payload)
        r.raise_for_status()
        return r.json()["conversation_id"]

async def send_to_conversation(conv_id: str, user_text: str) -> str:
    payload = {"role": "user", "content": user_text}
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(f"{PROXY_URL}/conversations/{conv_id}/messages", json=payload)
        r.raise_for_status()
        return r.json()["assistant_response"]

async def tts_to_file(text: str, voice: str = "en-US-AriaNeural") -> str:
    path = tempfile.mktemp(suffix=".mp3")
    communicate = edge_tts.Communicate(text, voice=voice)
    with open(path, "wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
    return path

async def ensure_voice(ctx: commands.Context) -> Optional[discord.VoiceClient]:
    if ctx.author.voice and ctx.author.voice.channel:
        channel = ctx.author.voice.channel
        if ctx.voice_client and ctx.voice_client.channel.id == channel.id:
            return ctx.voice_client
        if ctx.voice_client:
            await ctx.voice_client.move_to(channel)
        else:
            return await channel.connect(reconnect=True, timeout=30.0)
        return ctx.voice_client
    else:
        await ctx.send("Join a voice channel first.")
        return None

@bot.command(help="Join your current voice channel")
async def join(ctx: commands.Context):
    vc = await ensure_voice(ctx)
    if vc:
        await ctx.send(f"Joined: {vc.channel.name}")

@bot.command(help="Leave voice channel")
async def leave(ctx: commands.Context):
    if ctx.voice_client:
        await ctx.voice_client.disconnect(force=True)
        await ctx.send("Left voice channel.")
    else:
        await ctx.send("Not connected.")

@bot.command(help="Speak with the LLM and play the reply in voice: !vchat your message")
async def vchat(ctx: commands.Context, *, message: str):
    vc = await ensure_voice(ctx)
    if not vc:
        return

    conv = conversations.get(ctx.channel.id)
    if conv is None:
        try:
            conv_id = await create_conversation(DEFAULT_INSTRUCTIONS)
            conversations[ctx.channel.id] = {"conv_id": conv_id, "instructions": DEFAULT_INSTRUCTIONS}
            await ctx.send("(New conversation created.)")
        except Exception as e:
            await ctx.send(f"Error creating conversation: {e}")
            return

    conv_id = conversations[ctx.channel.id]["conv_id"]

    await ctx.trigger_typing()
    try:
        reply = await send_to_conversation(conv_id, message)
    except Exception as e:
        await ctx.send(f"Proxy error: {e}")
        return

    # Send text in chat too (optional)
    await ctx.send(reply[:1900])

    # TTS and play in voice
    try:
        mp3_path = await tts_to_file(reply)
        # FFmpeg decodes the MP3 for Discord
        source = discord.FFmpegPCMAudio(executable="ffmpeg", source=mp3_path)
        if vc.is_playing():
            vc.stop()
        vc.play(source)
    except Exception as e:
        await ctx.send(f"TTS/Playback error: {e}")

@bot.command(help='Set persona and reset conversation: !persona "You are ..."')
async def persona(ctx: commands.Context, *, instructions: str):
    try:
        conv_id = await create_conversation(instructions)
        conversations[ctx.channel.id] = {"conv_id": conv_id, "instructions": instructions}
        await ctx.send("Persona updated and new conversation started.")
    except Exception as e:
        await ctx.send(f"Error setting persona: {e}")

if __name__ == "__main__":
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise SystemExit("Set DISCORD_TOKEN env var.")
    bot.run(token)