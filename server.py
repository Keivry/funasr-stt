#!/usr/bin/env python3
"""FunASR OpenAI-Compatible API Server — Pure CPU version."""

import argparse
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FunASR STT Server")

DEVICE = "cpu"
MODEL_REGISTRY = {}

MODEL_CONFIGS = {
    "sensevoice": {
        "model": "iic/SenseVoiceSmall",
        "vad_model": "fsmn-vad",
        "punc_model": "ct-punc",
        "vad_kwargs": {"max_single_segment_time": 30000},
    },
    "paraformer": {
        "model": "paraformer-zh",
        "vad_model": "fsmn-vad",
        "punc_model": "ct-punc",
    },
    "paraformer-en": {
        "model": "paraformer-en",
        "vad_model": "fsmn-vad",
    },
    "fun-asr-nano": {
        "model": "FunAudioLLM/Fun-ASR-Nano-2512",
        "vad_model": "fsmn-vad",
        "vad_kwargs": {"max_single_segment_time": 30000},
        "trust_remote_code": True,
    },
}


def clean_text(text: str) -> str:
    return re.sub(r'<\|[^|]*\|>', '', text).strip()


def load_model(model_name: str):
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}"
        )
    from funasr import AutoModel

    cfg = MODEL_CONFIGS[model_name].copy()
    cfg["device"] = DEVICE
    cfg["disable_update"] = True
    logger.info("Loading model '%s' on device '%s'...", model_name, DEVICE)
    model = AutoModel(**cfg)
    MODEL_REGISTRY[model_name] = model
    logger.info("Model '%s' loaded successfully.", model_name)
    return model


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "loaded_models": list(MODEL_REGISTRY.keys()),
        "available_models": list(MODEL_CONFIGS.keys()),
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "ready": name in MODEL_REGISTRY}
            for name in MODEL_CONFIGS
        ],
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(default="sensevoice"),
    language: Optional[str] = Form(default=None),
    response_format: Optional[str] = Form(default="json"),
):
    if model not in MODEL_CONFIGS:
        raise HTTPException(
            400, f"Unknown model '{model}'. Available: {list(MODEL_CONFIGS.keys())}"
        )

    asr_model = load_model(model)

    suffix = Path(file.filename or "audio.wav").suffix
    if not suffix:
        suffix = ".wav"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()

        kwargs = {"batch_size": 1}
        if language:
            kwargs["language"] = language
        result = asr_model.generate(input=tmp.name, **kwargs)
        text = clean_text(result[0]["text"])

        if response_format == "verbose_json":
            segments = []
            for seg in result[0].get("sentence_info", []):
                segments.append(
                    {
                        "start": seg.get("start", 0) / 1000.0,
                        "end": seg.get("end", 0) / 1000.0,
                        "text": clean_text(seg.get("text", "")),
                        "speaker": seg.get("spk"),
                    }
                )
            return JSONResponse(
                {
                    "text": text,
                    "segments": segments,
                    "model": model,
                    "device": DEVICE,
                }
            )
        else:
            return JSONResponse({"text": text})
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(500, f"Transcription failed: {e}")
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", default="sensevoice")
    args = parser.parse_args()

    global DEVICE
    DEVICE = args.device
    logger.info(
        "Starting FunASR server on %s:%s (device=%s, model=%s)",
        args.host,
        args.port,
        args.device,
        args.model,
    )
    load_model(args.model)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
