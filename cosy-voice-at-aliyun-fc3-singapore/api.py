import sys, os
from typing import Optional
sys.path.append('third_party/Matcha-TTS')
from fastapi import FastAPI, WebSocket
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import warnings
import onnxruntime as ort
import io
import base64
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pathlib

app = FastAPI()

# Add static files mounting
current_dir = pathlib.Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(current_dir)), name="static")

# Configure ONNX Runtime global settings
ort.set_default_logger_severity(3)

# Set up providers
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']

# Initialize CosyVoice2 globally
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False)
prompt_speech_16k = load_wav('zero_shot_prompt.wav', 16000)
prompt = "希望你以后能够做的比我还好呦。"

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

@app.websocket("/tts")
async def tts_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        # Receive the text from the client
        data = await websocket.receive_text()
        text = data

        # Generate and stream audio chunks
        print(f"Starting TTS generation for text: {text}")
        for i, result in enumerate(cosyvoice.inference_zero_shot(text, prompt, prompt_speech_16k, stream=True)):
            print(f"Generating chunk {i}...")
            
            # Convert audio tensor to bytes
            buffer = io.BytesIO()
            print(f"Converting chunk {i} to WAV format...")
            torchaudio.save(buffer, result['tts_speech'], cosyvoice.sample_rate, format="wav")
            audio_bytes = buffer.getvalue()
            print(f"Chunk {i} size: {len(audio_bytes)} bytes")
            
            # Encode to base64 for sending over WebSocket
            print(f"Encoding chunk {i} to base64...")
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Send the audio chunk
            print(f"Sending chunk {i} over WebSocket...")
            print(f"Audio base64 data for chunk {i}: {audio_base64[:100]}...") # Show first 100 chars
            await websocket.send_json({
                "chunk_id": i,
                "audio_data": audio_base64
            })
            print(f"Successfully sent chunk {i}")
        print("TTS generation and streaming completed")
            
    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()

@app.get("/", response_class=HTMLResponse)
async def root():
    return open("index.html").read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=50000)