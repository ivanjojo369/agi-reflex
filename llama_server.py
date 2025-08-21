# server_llama.py
# FastAPI + llama-cpp-python (OpenChat-3.5-1210), baja latencia y memoria "lite"
from typing import Any, Dict, List, Optional, Generator
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import os

# ====== CONFIG ======
MODEL_PATH = r"C:\Models\openchat-3.5-1210.Q4_K_M.gguf"  # <-- AJUSTA ESTA RUTA
N_CTX = 4096
N_GPU_LAYERS = -1        # -1 = offload máximo si compilaste con CUDA; en CPU usa 0
N_THREADS = max(2, os.cpu_count() - 1)
N_BATCH = 512            # sube/baja si ves cortes; 512 va bien en CUDA/CPU modernas

DEFAULT_STOP = ["<|end_of_turn|>", "</s>"]
DEFAULT_PARAMS = dict(
    max_new_tokens=150,
    temperature=0.6,
    top_p=0.9,
    top_k=40,
    repeat_penalty=1.08,
    min_p=0.05,          # soportado por llama.cpp recientes
    stream=False
)

# ====== LLAMA INIT ======
LLM = None
LOAD_ERROR: Optional[str] = None
try:
    from llama_cpp import Llama
    LLM = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        n_threads=N_THREADS,
        n_batch=N_BATCH,
        use_mlock=True,
        logits_all=False,
        vocab_only=False,
        seed=0
    )
except Exception as e:
    LOAD_ERROR = f"{type(e).__name__}: {e}"

# ====== FASTAPI ======
app = FastAPI(title="AGI Local – OpenChat 3.5 (llama.cpp)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Memoria "lite" en RAM (simple)
STORE: List[Dict[str, Any]] = []

# ====== UTILS ======
def cut_at_stop(text: str, stops: List[str]) -> str:
    for s in stops:
        i = text.find(s)
        if i >= 0:
            return text[:i].rstrip()
    return text.rstrip()

def to_chatml(messages: List[Dict[str, str]]) -> str:
    """
    Construye prompt ChatML compacto para OpenChat, con turnos y tag final <|assistant|>.
    No cerrar con <|end_of_turn|> al final: dejamos que el modelo genere hasta stop.
    """
    parts: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        tag = "user"
        if role == "system": tag = "system"
        elif role == "assistant": tag = "assistant"
        parts.append(f"<|{tag}|>\n{content}\n<|end_of_turn|>")
    parts.append("<|assistant|>\n")   # listo para generar
    return "\n".join(parts)

def norm_params(p: Dict[str, Any]) -> Dict[str, Any]:
    out = DEFAULT_PARAMS.copy()
    if not p: return out
    for k in out:
        if k in p: out[k] = p[k]
    # limitar para latencia
    out["max_new_tokens"] = int(max(16, min(240, out["max_new_tokens"])))
    out["temperature"] = float(max(0.0, min(1.5, out["temperature"])))
    out["top_p"] = float(max(0.05, min(1.0, out["top_p"])))
    out["top_k"] = int(max(1, min(100, out["top_k"])))
    out["repeat_penalty"] = float(max(1.0, min(1.5, out["repeat_penalty"])))
    out["min_p"] = float(max(0.0, min(0.5, out["min_p"])))
    return out

# ====== ROUTES ======
@app.get("/")
def health():
    return {"ok": True, "model_loaded": LOAD_ERROR is None}

@app.post("/memory/semantic/upsert")
def mem_upsert(payload: Dict[str, Any] = Body(...)):
    facts = payload.get("facts", [])
    for f in facts:
        if f.get("text"): STORE.append(f)
    return {"upserted": len(facts)}

@app.post("/memory/semantic/search")
def mem_search(payload: Dict[str, Any] = Body(...)):
    q = (payload.get("q") or "").lower(); k = int(payload.get("k", 3))
    scored = []
    for f in STORE:
        t = f.get("text","")
        score = t.lower().count(q) if q else 1
        scored.append((score, f))
    scored.sort(key=lambda x: x[0], reverse=True)
    return {"results": [x[1] for x in scored[:k]]}

@app.post("/chat")
def chat(payload: Dict[str, Any] = Body(...)):
    if LOAD_ERROR:
        return JSONResponse({"error":"model_not_loaded", "detail":LOAD_ERROR}, status_code=503)

    messages = payload.get("messages") or []
    params = norm_params(payload.get("params") or {})
    stops: List[str] = payload.get("params", {}).get("stop") or DEFAULT_STOP
    stream: bool = bool(params.get("stream", False))

    prompt = to_chatml(messages)

    # Llama.cpp expects max_tokens (tokens a generar)
    kwargs = dict(
        prompt=prompt,
        max_tokens=int(params["max_new_tokens"]),
        temperature=float(params["temperature"]),
        top_p=float(params["top_p"]),
        top_k=int(params["top_k"]),
        repeat_penalty=float(params["repeat_penalty"]),
        min_p=float(params["min_p"]),
        stop=stops,
        stream=stream
    )

    if stream:
        def gen() -> Generator[bytes, None, None]:
            try:
                for ev in LLM.create_completion(**kwargs):
                    # cada 'ev' tiene choices[0].text
                    piece = ev["choices"][0]["text"]
                    yield piece.encode("utf-8")
            except Exception as e:
                yield f"\n[stream_error] {e}".encode("utf-8")
        return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")

    # no stream
    out = LLM.create_completion(**kwargs)
    text = out["choices"][0]["text"]
    text = cut_at_stop(text, stops)
    return {"text": text}
