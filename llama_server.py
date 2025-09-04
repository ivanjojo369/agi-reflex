# -*- coding: utf-8 -*-
"""
Servidor FastAPI para modelo local (llama.cpp) con seguridad y memoria semántica.

Endpoints:
- GET  /                            -> health/info
- POST /chat                        -> chat (messages+params)  [stream | no-stream]
- POST /memory/semantic/upsert      -> upsert de memoria semántica
- POST /memory/semantic/search      -> búsqueda semántica

Seguridad:
- Header  X-API-Key: <clave>
- Ó  Authorization: Bearer <clave>
Se activa si REQUIRE_API_KEY=1 (o "true") en el proceso del *server*.

Config:
- Lee primero server_config.json (opcional) y luego variables de entorno:
  MODEL_PATH, N_CTX, N_THREADS, N_BATCH, N_GPU_LAYERS
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("llama_server")

# -----------------------------
# Seguridad (Swagger Authorize)
# -----------------------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)

def require_key() -> bool:
    return str(os.getenv("REQUIRE_API_KEY", "0")).lower() in ("1", "true", "yes")

async def check_api_key(
    api_key: Optional[str] = Security(api_key_header),
    bearer: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> None:
    """Valida API key si REQUIRE_API_KEY=1. Usa X-API-Key o Bearer."""
    if not require_key():
        return

    expected = os.getenv("DEMO_KEY") or os.getenv("API_KEY")
    provided = api_key or (bearer.credentials if bearer else None)

    if not expected:
        raise HTTPException(status_code=500, detail="Server misconfigured: DEMO_KEY not set")

    if not provided or provided != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

# -----------------------------
# Carga de configuración
# -----------------------------
def load_server_config() -> Dict[str, Any]:
    """Mezcla server_config.json con variables de entorno."""
    cfg_path = Path("server_config.json")
    file_cfg: Dict[str, Any] = {}
    if cfg_path.exists():
        try:
            file_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("No se pudo leer server_config.json: %s", e)

    def env_int(name: str, default: Optional[int]) -> Optional[int]:
        v = os.getenv(name)
        return int(v) if v not in (None, "") else default

    def env_str(name: str, default: Optional[str]) -> Optional[str]:
        v = os.getenv(name)
        return v if v not in (None, "") else default

    cfg = {
        "MODEL_PATH": env_str("MODEL_PATH", file_cfg.get("MODEL_PATH")),
        "N_CTX": env_int("N_CTX", file_cfg.get("N_CTX", 4096)),
        "N_THREADS": env_int("N_THREADS", file_cfg.get("N_THREADS", 8)),
        "N_BATCH": env_int("N_BATCH", file_cfg.get("N_BATCH", 192)),
        "N_GPU_LAYERS": env_int("N_GPU_LAYERS", file_cfg.get("N_GPU_LAYERS", 0)),
    }
    return cfg

CONFIG = load_server_config()

# -----------------------------
# Modelo (adapter)
# -----------------------------
try:
    # Usa tu adapter local
    from adapters.llama_cpp_adapter import LlamaCppChat  # type: ignore
except Exception as e:  # pragma: no cover
    log.error("No se pudo importar adapters.llama_cpp_adapter: %s", e)
    LlamaCppChat = None  # type: ignore

chat_model = None  # instancia global

def _build_model() -> Optional[Any]:
    mp = CONFIG.get("MODEL_PATH")
    if not mp:
        log.warning("No se definió MODEL_PATH. El server inicia, pero /chat devolverá 503.")
        return None

    if LlamaCppChat is None:
        log.error("Adapter LlamaCppChat no disponible.")
        return None

    try:
        import inspect

        # Descubre qué nombres acepta el __init__ de tu adapter
        init_params = set(inspect.signature(LlamaCppChat.__init__).parameters.keys())

        def choose_name(*candidates: str) -> Optional[str]:
            # devuelve el primer nombre que exista en el constructor
            for c in candidates:
                if c in init_params:
                    return c
            return None

        # Construye kwargs usando sinónimos comunes
        kwargs = {}
        # model path
        name_model = choose_name("model_path", "model", "path")
        if not name_model:
            log.error("El adapter no acepta 'model_path'/'model'/'path' en el __init__.")
            return None
        kwargs[name_model] = mp

        # ctx
        name_ctx = choose_name("n_ctx", "ctx", "context_length", "ctx_size", "n_ctx_tokens")
        if name_ctx:
            kwargs[name_ctx] = CONFIG["N_CTX"]

        # threads
        name_thr = choose_name("n_threads", "threads", "num_threads")
        if name_thr:
            kwargs[name_thr] = CONFIG["N_THREADS"]

        # batch
        name_bat = choose_name("n_batch", "batch", "batch_size")
        if name_bat:
            kwargs[name_bat] = CONFIG["N_BATCH"]

        # gpu layers
        name_gpu = choose_name("n_gpu_layers", "gpu_layers", "n_gpu")
        if name_gpu:
            kwargs[name_gpu] = CONFIG["N_GPU_LAYERS"]

        log.info(
            "Cargando modelo: %s (ctx=%s, threads=%s, batch=%s, gpu_layers=%s) "
            "→ mapeo __init__: %s",
            mp, CONFIG["N_CTX"], CONFIG["N_THREADS"], CONFIG["N_BATCH"], CONFIG["N_GPU_LAYERS"],
            {k: kwargs[k] for k in kwargs if k != name_model},
        )

        return LlamaCppChat(**kwargs)

    except Exception as e:
        log.error("No se pudo cargar el modelo: %s", e)
        return None

# -----------------------------
# Memoria semántica (opcional)
# -----------------------------
class NoopMemory:
    def upsert(self, texts: List[str], namespace: Optional[str] = None, metadatas: Optional[List[Dict]] = None) -> Dict:
        return {"ok": True, "count": len(texts), "detail": "noop"}

    def search(self, query: str, k: int = 5, namespace: Optional[str] = None) -> Dict:
        return {"ok": True, "results": []}

try:
    from memory.semantic_memory import SemanticMemory  # type: ignore
    memory = SemanticMemory()
    log.info("Memoria semántica: OK")
except Exception as e:  # pragma: no cover
    log.warning("Memoria semántica no disponible (%s). Usando Noop.", e)
    memory = NoopMemory()

# -----------------------------
# Pydantic models
# -----------------------------
class Message(BaseModel):
    role: str = Field(..., description="system|user|assistant")
    content: str

class ChatParams(BaseModel):
    max_new_tokens: int = 128
    temperature: float = 0.2
    top_p: float = 0.95
    stop: Optional[List[str]] = None
    stream: bool = False

class ChatRequest(BaseModel):
    messages: List[Message]
    params: ChatParams = ChatParams()

class UpsertRequest(BaseModel):
    texts: List[str]
    namespace: Optional[str] = None
    metadatas: Optional[List[Dict[str, Any]]] = None

class SearchRequest(BaseModel):
    query: str
    k: int = 5
    namespace: Optional[str] = None

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="Local Llama Server",
    version="1.1.0",
    description="Servidor para modelo local (llama.cpp) con API Key opcional y memoria semántica.",
)

# Construye el modelo al importar el módulo (inicio del server)
chat_model = _build_model()

@app.get("/")
async def health() -> Dict[str, Any]:
    """Estado del servidor y configuración efectiva (sin exponer claves)."""
    return {
        "ok": True,
        "model_loaded": bool(chat_model),
        "model_path": CONFIG.get("MODEL_PATH"),
        "require_api_key": require_key(),
        "env": {
            "N_CTX": str(CONFIG.get("N_CTX")),
            "N_THREADS": str(CONFIG.get("N_THREADS")),
            "N_BATCH": str(CONFIG.get("N_BATCH")),
            "N_GPU_LAYERS": str(CONFIG.get("N_GPU_LAYERS")),
        },
        "tips": (
            "Si model_loaded=false, coloca tu .gguf en ./models o define MODEL_PATH "
            "en server_config.json o variable de entorno."
        ),
    }

@app.post("/chat")
async def chat(req: ChatRequest, _=Security(check_api_key)):
    """Chat sencillo con/no streaming. Espera messages (role/content) y params."""
    if chat_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Normaliza messages → lista de dicts
    messages = [m.model_dump() for m in req.messages]
    p = req.params

    # kwargs compatibles con el adapter
    kwargs = {
        "max_new_tokens": p.max_new_tokens,
        "temperature": p.temperature,
        "top_p": p.top_p,
        "stop": p.stop or [],
    }

    # Streaming (si el adapter lo soporta)
    if p.stream:
        if hasattr(chat_model, "chat_stream"):
            gen = chat_model.chat_stream(messages, **kwargs)  # type: ignore
        elif hasattr(chat_model, "stream_chat"):
            gen = chat_model.stream_chat(messages, **kwargs)  # type: ignore
        elif hasattr(chat_model, "stream"):
            gen = chat_model.stream(messages, **kwargs)  # type: ignore
        else:
            def gen() -> Iterable[str]:
                res = chat_model.chat(messages, **kwargs)  # type: ignore
                if isinstance(res, dict) and "text" in res:
                    yield str(res["text"])
                else:
                    yield str(res)

        def _iter() -> Generator[bytes, None, None]:
            for chunk in gen:
                yield (chunk if isinstance(chunk, str) else str(chunk)).encode("utf-8")

        return StreamingResponse(_iter(), media_type="text/plain; charset=utf-8")

    # No-stream
    if hasattr(chat_model, "chat"):
        res = chat_model.chat(messages, **kwargs)  # type: ignore
    else:
        res = chat_model.generate(messages, **kwargs)  # type: ignore

    # Normaliza salida para el agente
    out_text = res.get("text") if isinstance(res, dict) else str(res)
    return JSONResponse({"output": {"text": out_text}})

@app.post("/memory/semantic/upsert")
async def mem_upsert(req: UpsertRequest, _=Security(check_api_key)):
    try:
        res = memory.upsert(req.texts, namespace=req.namespace, metadatas=req.metadatas)
        return JSONResponse(res)
    except Exception as e:
        log.error("Error en upsert: %s", e)
        raise HTTPException(status_code=500, detail="Memory upsert failed")

@app.post("/memory/semantic/search")
async def mem_search(req: SearchRequest, _=Security(check_api_key)):
    try:
        res = memory.search(req.query, k=req.k, namespace=req.namespace)
        return JSONResponse(res)
    except Exception as e:
        log.error("Error en search: %s", e)
        raise HTTPException(status_code=500, detail="Memory search failed")

# -----------------------------
# Main (dev only)
# -----------------------------
if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run("llama_server:app", host="127.0.0.1", port=8010, reload=False)
