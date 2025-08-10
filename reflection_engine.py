# agents/reflection_engine.py
# Motor de reflexión con anti-eco, anti-repetición y contexto (slots + historial + semántica).
from __future__ import annotations
from typing import Any, List, Optional
import re

# Reglas permanentes (estilo + guardas)
SYSTEM_STYLE_TPL = (
    "Eres un asistente útil, claro y conversacional. "
    "No imprimas encabezados, plantillas, etiquetas ni marcadores del sistema. "
    "No repitas ni cites literalmente el texto del usuario. "
    "Entrega UNA sola respuesta, concreta y enfocada en la petición. "
    "Si hay memoria relevante, intégrala con naturalidad."
)

# Marcadores/encabezados que algunos modelos tienden a imprimir
_SYS_MARKERS = [
    "[SISTEMA]", "[SYSTEM]", "[CONTEXTO]", "[CONTEXT]", "[USUARIO]", "[USER]",
    "[ASISTENTE]", "[ASSISTANT]", "[RESPONDER]", "[RESPONSE]",
    "<think>", "</think>", "[INSTRUCCIONES]", "INSTRUCCIONES", "INSTRUCTIONS"
]

_META_HEADERS = [
    "MENSAJE DEL USUARIO", "MENSAJE DEL ASISTENTE",
    "RESPONDER", "RESPONSE", "INSTRUCCIONES", "INSTRUCTIONS"
]

# ---------- Utilidades de limpieza ----------
def _strip_system_markers(text: str) -> str:
    t = text or ""
    for m in _SYS_MARKERS:
        t = t.replace(m, "")
    t = re.sub(r"(?:^|\n)[\-\*\•]\s*$", "", t.strip())   # bullets huérfanos
    t = re.sub(r"(?:\n\s*){3,}", "\n\n", t)              # colapsa saltos extra
    return t.strip()

def _strip_meta_headers(text: str) -> str:
    if not text: return ""
    pat = r"^\s*(?:" + "|".join(map(re.escape, _META_HEADERS)) + r")\s*:?\s*$"
    return re.sub(pat, "", text, flags=re.IGNORECASE | re.MULTILINE).strip()

def _strip_instruction_lines(text: str) -> str:
    if not text: return ""
    out = []
    for ln in text.splitlines():
        low = ln.lower().strip()
        if low.startswith("- no ") or "no imprimas" in low or "no cites" in low:
            continue
        if low in ("[responder]", "[response]"):
            continue
        out.append(ln)
    return "\n".join(out).strip()

def _dedupe_lines(text: str) -> str:
    if not text: return text
    out = []
    for ln in text.splitlines():
        if not out or ln.strip() != out[-1].strip():
            out.append(ln.rstrip())
    return "\n".join(out).strip()

def _remove_user_echo(text: str, user_text: str) -> str:
    if not text or not user_text: return text or ""
    t = text.replace(user_text.strip(), "")
    ut = re.escape(user_text.strip().rstrip("?.!"))
    t = re.sub(rf"\b{ut}\b[?.!…]*", "", t, flags=re.IGNORECASE)
    return t.strip()

def _too_similar(a: str, b: str, threshold: float = 0.90) -> bool:
    if not a or not b: return False
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb: return False
    j = len(ta & tb) / max(1, len(ta | tb))
    return j >= threshold

# ---------- ReflectionEngine ----------
class ReflectionEngine:
    def __init__(self,
                 style: str = "helpful, reflective, analytical, empathetic",
                 max_steps: int = 2,
                 enable_memory_enrichment: bool = True):
        self.style = style
        self.max_steps = max_steps
        self.enable_memory_enrichment = enable_memory_enrichment
        self.sem = None  # se engancha desde el initializer si existe

    # Contexto breve desde memoria (slots + últimos turnos)
    def _format_context(self, memory) -> str:
        if not self.enable_memory_enrichment:
            return ""
        parts: List[str] = []
        # Slots resumidos
        try:
            slots = memory.summary_text()
            if slots:
                parts.append("SLOTS\n" + slots)
        except Exception:
            pass
        # Historial reciente (máx. 6)
        try:
            recent = memory.recall_recent(6)
            if recent:
                lines = []
                for r in recent:
                    role = getattr(r, "role", "user").lower()
                    who = "Usuario" if role == "user" else "Asistente"
                    content = getattr(r, "content", str(r)).strip()
                    if len(content) > 220:
                        content = content[:220] + "…"
                    lines.append(f"- {who}: {content}")
                parts.append("HISTORIAL\n" + "\n".join(lines))
        except Exception:
            pass
        return "\n\n".join(parts).strip()

    # Contexto semántico si hay memoria vectorial “lite”
    def _semantic_context(self, query: str) -> str:
        if not self.sem:
            return ""
        try:
            return self.sem.context_block(query, k=3, max_chars=900) or ""
        except Exception:
            return ""

    # Construcción del prompt (minimizando plantillas)
    def build_prompt(self, user_message: str, memory) -> str:
        blocks = []
        ctx = self._format_context(memory)
        if ctx: blocks.append(ctx)
        sem = self._semantic_context(user_message)
        if sem: blocks.append(sem)

        context_text = ""
        if blocks:
            context_text = "CONTEXTO\n" + "\n\n".join(blocks) + "\n\n"

        # No pongo "Instrucciones:" para reducir eco; las reglas van en SYSTEM_STYLE_TPL
        prompt = (
            SYSTEM_STYLE_TPL + "\n\n" +
            context_text +
            "MENSAJE DEL USUARIO\n" + user_message.strip()
        )
        return prompt

    # Llamada al modelo para quienes prefieran usar la clase directamente
    def call_model(self, model: Any, prompt: str,
                   temperature: float = 0.55, max_tokens: int = 256,
                   stop: Optional[list[str]] = None) -> str:
        if hasattr(model, "generate"):
            return model.generate(prompt=prompt, temperature=temperature,
                                  max_tokens=max_tokens, stop=stop)
        if hasattr(model, "chat"):
            messages = [{"role": "system", "content": SYSTEM_STYLE_TPL},
                        {"role": "user", "content": prompt}]
            return model.chat(messages=messages, temperature=temperature,
                              max_tokens=max_tokens, stop=stop)
        if hasattr(model, "completion"):
            return model.completion(prompt=prompt, temperature=temperature,
                                    max_tokens=max_tokens, stop=stop)
        raise AttributeError("El adaptador de modelo no expone generate/chat/completion.")

    # Post-procesado: limpia y evita eco/repetición
    def postprocess(self, raw: str, user_text: str, memory) -> str:
        txt = raw or ""
        txt = _strip_system_markers(txt)
        txt = _strip_meta_headers(txt)
        txt = _strip_instruction_lines(txt)
        txt = _remove_user_echo(txt, user_text)
        txt = _dedupe_lines(txt)

        # Evita respuestas casi idénticas a previas del asistente
        try:
            recent = memory.recall_recent(4)
            recent_assist = [r.content for r in recent if getattr(r, "role", "") == "assistant"]
            for prev in recent_assist:
                if _too_similar(txt, prev, threshold=0.90):
                    txt = txt.split("\n")[0].strip()
                    break
        except Exception:
            pass

        if len(txt) > 2000:
            txt = txt[:2000].rstrip() + "…"
        return txt
