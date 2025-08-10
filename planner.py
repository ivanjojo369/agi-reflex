# planner/planner.py
from __future__ import annotations
import re
import json
from pathlib import Path
from typing import Dict, Any, Optional

RULES_PATH = Path(__file__).resolve().parent / "rules.json"

_INTENT_PATTERNS = [
    ("remember", r"\b(recuerda que|anota que|guarda que|me llamo|estoy en|vivo en|me gusta)\b"),
    ("recall",   r"\b(qué me anotaste|qué recuerdas|qué sabes de mí|cómo me llamo|dónde estoy|mis preferencias)\b"),
    ("math",     r"\b(calcula|cuánto es|resuelve|^[-+*/()\d\.\s^]+$)"),
    ("datetime", r"\b(qué hora es|fecha de hoy|hoy es|mañana es|ayer fue)\b"),
    ("tasklist", r"\b(lista de tareas|to-?do|pendientes)\b"),
    ("smalltalk",r".*"),  # fallback
]

def _load_rules() -> Dict[str, Any]:
    if RULES_PATH.exists():
        try:
            return json.loads(RULES_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    # fallback mínimo si el rules.json no está disponible
    return {
        "remember": {"next": "memory.update"},
        "recall": {"next": "memory.report"},
        "math": {"next": "tool.calculator"},
        "datetime": {"next": "tool.datetime"},
        "tasklist": {"next": "tool.tasklist_stub"},
        "smalltalk": {"next": "llm.reply"}
    }

class Planner:
    def __init__(self):
        self.rules = _load_rules()

    def detect(self, text: str, memory=None) -> Dict[str, Any]:
        t = (text or "").strip().lower()
        intent = "smalltalk"
        slots: Dict[str, Any] = {}

        for name, pat in _INTENT_PATTERNS:
            if re.search(pat, t):
                intent = name
                break

        # extracción simple de slots
        if intent == "remember":
            # nombre
            m = re.search(r"\bme llamo\s+([a-záéíóúñ]+)", t)
            if m: slots["name"] = m.group(1).strip().title()
            # ubicación
            m = re.search(r"\b(estoy en|vivo en)\s+(.+)", t)
            if m: slots["location"] = m.group(2).strip().rstrip(".")
            # preferencias
            m = re.search(r"\bme gusta[n]?\s+(.+)", t)
            if m: slots.setdefault("preferences", []).append(m.group(1).strip().rstrip("."))

        return {
            "intent": intent,
            "slots": slots,
            "next": self.rules.get(intent, {}).get("next", "llm.reply")
        }
