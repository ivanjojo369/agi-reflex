# memory/context_memory.py
# Memoria por "slots" (nombre, ubicación, preferencias) + historial reciente.
# Persistencia simple en JSON. Sin FAISS (Hito Fusión 1).

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

DEFAULT_PERSIST_PATH = Path("chroma_db") / "slot_memory.json"

@dataclass
class SlotState:
    name: Optional[str] = None
    location: Optional[str] = None
    preferences: List[str] = field(default_factory=list)
    facts: Dict[str, str] = field(default_factory=dict)

@dataclass
class MessageRecord:
    role: str
    content: str
    ts: float

class ContextMemory:
    """
    Memoria liviana: slots + historial reciente.
    """
    def __init__(
        self,
        persist_path: Path | str = DEFAULT_PERSIST_PATH,
        save_to_disk: bool = True,
        max_history: int = 40
    ):
        self.persist_path = Path(persist_path)
        self.save_to_disk = bool(save_to_disk)
        self.max_history = int(max_history)
        self.slots = SlotState()
        self.history: List[MessageRecord] = []

        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_if_exists()

    # -------- Persistencia --------

    def _load_if_exists(self):
        if self.persist_path.exists():
            try:
                data = json.loads(self.persist_path.read_text(encoding="utf-8"))
                s = data.get("slots", {})
                self.slots = SlotState(
                    name=s.get("name"),
                    location=s.get("location"),
                    preferences=list(s.get("preferences", [])),
                    facts=dict(s.get("facts", {})),
                )
                self.history = [
                    MessageRecord(**m) for m in data.get("history", [])
                ]
            except Exception:
                # Archivo corrupto: arrancar limpio sin romper
                self.slots = SlotState()
                self.history = []

    def persist(self):
        if not self.save_to_disk:
            return
        data = {
            "slots": asdict(self.slots),
            "history": [asdict(h) for h in self.history[-self.max_history:]],
        }
        self.persist_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # -------- Slots API --------

    def update_slot(self, slot: str, value: str):
        slot = slot.strip().lower()
        if slot == "name":
            self.slots.name = value.strip()
        elif slot == "location":
            self.slots.location = value.strip()
        else:
            self.slots.facts[slot] = value.strip()
        self.persist()

    def get_slot(self, slot: str) -> Optional[str]:
        slot = slot.strip().lower()
        if slot == "name":
            return self.slots.name
        if slot == "location":
            return self.slots.location
        return self.slots.facts.get(slot)

    def add_preference(self, pref: str):
        pref = pref.strip()
        if pref and pref not in self.slots.preferences:
            self.slots.preferences.append(pref)
            self.persist()

    def get_preferences(self) -> List[str]:
        return list(self.slots.preferences)

    def forget_slot(self, slot: str):
        slot = slot.strip().lower()
        if slot == "name":
            self.slots.name = None
        elif slot == "location":
            self.slots.location = None
        else:
            self.slots.facts.pop(slot, None)
        self.persist()

    # -------- Historial --------

    def add_history(self, role: str, content: str):
        self.history.append(MessageRecord(role=role, content=content, ts=time.time()))
        # recorte suave
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        # no persistimos cada token; se persiste en eventos clave
        # pero este recorte sí conviene guardar
        self.persist()

    def recall_recent(self, k: int = 6) -> List[MessageRecord]:
        return self.history[-k:] if k > 0 else []

    # -------- Utilidades --------

    def summary_text(self) -> str:
        """
        Sumariza el estado conocido para enriquecer la reflexión.
        """
        parts = []
        if self.slots.name:
            parts.append(f"Nombre del usuario: {self.slots.name}.")
        if self.slots.location:
            parts.append(f"Ubicación del usuario: {self.slots.location}.")
        if self.slots.preferences:
            prefs = ", ".join(self.slots.preferences[:8])
            parts.append(f"Preferencias del usuario: {prefs}.")
        if self.slots.facts:
            facts = "; ".join([f"{k}: {v}" for k, v in list(self.slots.facts.items())[:8]])
            parts.append(f"Hechos: {facts}.")
        return " ".join(parts) if parts else "Sin datos de slots aún."
