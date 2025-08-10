# agi_interface.py
from __future__ import annotations
from typing import Any, Dict, Optional
import re

# Stopwords típicos para modelos tipo OpenChat/llama.cpp
STOP_DEFAULT = ["<|end_of_turn|>", "</s>", "Usuario:", "User:", "\nUsuario:", "\nUser:"]

def _dedupe_blocks(text: str) -> str:
    if not text:
        return text
    # elimina líneas duplicadas consecutivas
    lines = [ln.rstrip() for ln in text.splitlines()]
    out = []
    for ln in lines:
        if not out or ln != out[-1]:
            out.append(ln)
    text = "\n".join(out)
    # colapsa saltos extra
    text = re.sub(r"(?:\n\s*){3,}", "\n\n", text)
    return text.strip()

def _sanitize(text: str, user_text: str = "") -> str:
    t = (text or "").strip()
    for b in ("[RESPONDER]", "[RESPONSE]", "Usuario:", "Asistente:", "<think>", "</think>"):
        t = t.replace(b, "")
    if user_text:
        # evita eco literal
        t = t.replace(user_text.strip(), "")
    return _dedupe_blocks(t)[:2000].strip()

def _max_tokens_by_input(user_text: str) -> int:
    n = len(user_text or "")
    if n <= 40: return 96
    if n <= 160: return 192
    return 256

class AGIInterface:
    def __init__(self,
                 model: Any,
                 memory: Any,
                 planner: Any,
                 task_manager: Any,
                 reflection_engine: Any,
                 episodic: Optional[Any] = None,
                 reply_style: str = "conversational",
                 use_meta_agent: bool = True):
        self.model = model
        self.memory = memory
        self.planner = planner
        self.task_manager = task_manager
        self.re = reflection_engine
        self.episodic = episodic
        self.reply_style = reply_style
        self.use_meta_agent = use_meta_agent
        self.tools: Dict[str, Any] = {}

    def _log_episode(self, intent: str, user_text: str, reply: str):
        if self.episodic:
            try:
                self.episodic.add(intent=intent, text=user_text, reply=reply, tags=[intent])
            except Exception:
                pass

    def _tool(self, key: str, func: str, *args, **kwargs) -> str:
        try:
            mod = self.tools.get(key)
            if not mod: return "Herramienta no disponible."
            f = getattr(mod, func, None)
            if not f: return "Acción no disponible."
            return str(f(*args, **kwargs))
        except Exception as e:
            return f"Error de herramienta: {e}"

    def process_message(self, user_text: str) -> str:
        # 1) planner
        plan: Dict[str, Any] = self.planner.detect(user_text, memory=self.memory)
        intent = plan.get("intent", "smalltalk")
        next_step = plan.get("next", "llm.reply")
        slots = plan.get("slots", {})

        # 2) acciones sin LLM
        if next_step == "memory.update":
            if "name" in slots: self.memory.update_slot("name", slots["name"])
            if "location" in slots: self.memory.update_slot("location", slots["location"])
            for p in slots.get("preferences", []): self.memory.add_preference(p)
            self.memory.add_history("user", user_text); self.memory.persist()
            reply = "¡Hecho! Lo recordaré."
            self._log_episode(intent, user_text, reply)
            return reply

        if next_step == "memory.report":
            get = getattr(self.memory, "get_slot", lambda *_: None)
            name = get("name"); loc = get("location")
            prefs = getattr(self.memory, "get_preferences", lambda : [])() or []
            parts = []
            if name: parts.append(f"Nombre: {name}")
            if loc: parts.append(f"Ubicación: {loc}")
            if prefs: parts.append("Preferencias: " + ", ".join(prefs))
            if not parts: parts.append("Aún no tengo datos guardados.")
            self.memory.add_history("user", user_text); self.memory.persist()
            reply = " / ".join(parts)
            self._log_episode(intent, user_text, reply)
            return reply

        if next_step == "tool.calculator":
            expr = user_text.lower().replace("calcula", "").strip()
            out = self._tool("calculator", "calculate", expr)
            self.memory.add_history("user", user_text); self.memory.persist()
            self._log_episode(intent, user_text, out)
            return out

        if next_step == "tool.datetime":
            out = self._tool("datetime", "handle", user_text)
            self.memory.add_history("user", user_text); self.memory.persist()
            self._log_episode(intent, user_text, out)
            return out

        if next_step == "tool.tasklist_stub":
            self.memory.add_history("user", user_text); self.memory.persist()
            reply = "Ok, (stub) pronto habilitaré la lista de tareas."
            self._log_episode(intent, user_text, reply)
            return reply

        # 3) LLM (prompt del ReflectionEngine + stops + sanitizado)
        self.memory.add_history("user", user_text)
        prompt = self.re.build_prompt(user_text, self.memory)
        max_new = _max_tokens_by_input(user_text)

        try:
            raw = self.model.generate(
                prompt=prompt,
                temperature=0.50,
                max_tokens=max_new,
                top_p=0.92,
                repeat_penalty=1.15,
                stop=STOP_DEFAULT
            )
        except TypeError:
            # adaptadores sin los mismos kwargs
            raw = self.re.call_model(self.model, prompt, temperature=0.50,
                                     max_tokens=max_new, stop=STOP_DEFAULT)

        reply = _sanitize(raw, user_text)
        self.memory.add_history("assistant", reply); self.memory.persist()
        self._log_episode(intent, user_text, reply)
        return reply
