#!/usr/bin/env python3
"""
Standalone Kokoro TTS worker process.

This worker runs in complete isolation to avoid Metal threading conflicts.
It communicates via JSON over stdin/stdout.

Usage:
    python kokoro_worker.py

Commands:
    {"cmd": "init", "model": "prince-canuma/Kokoro-82M", "voice": "af_heart"}
    {"cmd": "generate", "text": "Hello world"}
"""

import sys
import json
import base64
import traceback
import numpy as np

# Add logging to worker - redirect to stderr to not interfere with JSON output
import logging
logging.basicConfig(level=logging.INFO, format='WORKER: %(message)s', stream=sys.stderr)

# Log Python version and path
print(f"WORKER: Python executable: {sys.executable}", file=sys.stderr)
print(f"WORKER: Python version: {sys.version}", file=sys.stderr)
print(f"WORKER: Python path: {sys.path}", file=sys.stderr)

try:
    # Suppress HuggingFace progress bars to stderr
    import os
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
    MLX_AVAILABLE = True
    print(f"WORKER: MLX imported successfully", file=sys.stderr)
except ImportError as e:
    MLX_AVAILABLE = False
    print(f"WORKER: Failed to import MLX: {e}", file=sys.stderr)
    print(f"WORKER: Import traceback: {traceback.format_exc()}", file=sys.stderr)


class Worker:
    def __init__(self):
        self.model = None
        self.voice = None
        
    def initialize(self, model_name, voice):
        if not MLX_AVAILABLE:
            return {"error": "MLX not available"}
        try:
            print(f"Initializing Kokoro model: {model_name} with voice: {voice}", file=sys.stderr)
            self.model = load_model(model_name)
            self.voice = voice
            # Test generation to ensure everything works
            print("Testing model with sample text...", file=sys.stderr)
            list(self.model.generate(text="test", voice=voice, speed=1.0))
            print("Model initialized successfully", file=sys.stderr)
            return {"success": True}
        except Exception as e:
            print(f"Initialization error: {e}", file=sys.stderr)
            return {"error": str(e)}
    
    def generate(self, text):
        try:
            print(f"Generating audio for text: {text[:50]}...", file=sys.stderr)
            if not self.model:
                return {"error": "Not initialized"}
            
            segments = []
            for result in self.model.generate(text=text, voice=self.voice, speed=1.0):
                # Convert MLX array to numpy immediately
                audio_data = np.array(result.audio, copy=True)
                # Debug output to stderr
                print(f"Generated segment shape: {audio_data.shape}, min: {audio_data.min():.4f}, max: {audio_data.max():.4f}", file=sys.stderr)
                segments.append(audio_data)
            
            if not segments:
                return {"error": "No audio"}
                
            # Concatenate all segments
            if len(segments) == 1:
                audio = segments[0]
            else:
                audio = np.concatenate(segments, axis=0)
            
            # Debug output to stderr
            print(f"Final audio shape: {audio.shape}, min: {audio.min():.4f}, max: {audio.max():.4f}", file=sys.stderr)
            
            # Check if audio is silent
            if np.max(np.abs(audio)) < 1e-6:
                return {"error": "Generated audio is silent"}
            
            # Convert to 16-bit PCM
            audio_int16 = (audio * 32767).astype(np.int16)
            audio_b64 = base64.b64encode(audio_int16.tobytes()).decode()
            
            print(f"Encoded audio: {len(audio_int16.tobytes())} bytes, base64 length: {len(audio_b64)}", file=sys.stderr)
            
            return {"success": True, "audio": audio_b64}
        except Exception as e:
            import traceback
            # Ensure error message is properly escaped for JSON
            error_msg = str(e).replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
            return {"error": error_msg}


def main():
    """Main worker loop - reads commands from stdin, writes responses to stdout."""
    worker = Worker()
    
    for line in sys.stdin:
        try:
            req = json.loads(line.strip())
            if req["cmd"] == "init":
                resp = worker.initialize(req["model"], req["voice"])
            elif req["cmd"] == "generate":
                resp = worker.generate(req["text"])
            else:
                resp = {"error": "Unknown command"}
            print(json.dumps(resp), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()