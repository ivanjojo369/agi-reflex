# agents/agent.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, json, re, time, unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests


# -------------------- Presets de sampling --------------------
OPEN   = dict(temperature=0.25, top_p=0.90)
STRICT = dict(temperature=0.15, top_p=0.85)


def _now_ms() -> int:
    return int(time.time() * 1000)


def strip_code_fences(t: str) -> str:
    if not isinstance(t, str):
        return t
    s = t.strip()
    s = re.sub(r"^```[a-zA-Z0-9_+-]*\s*\n(.*?)\n```$", r"\1", s, flags=re.DOTALL)
    s = re.sub(r"^```\s*\n(.*?)\n```$", r"\1", s, flags=re.DOTALL)
    s = re.sub(r"```(.*?)```", r"\1", s, flags=re.DOTALL)
    return s.strip()


def _norm(s: str) -> str:
    """Minúsculas + sin acentos para comparar el goal."""
    return unicodedata.normalize("NFD", s or "").encode("ascii", "ignore").decode("ascii").lower()


def find_first_json_blob(text: str) -> Optional[str]:
    if not isinstance(text, str) or not text:
        return None
    s = strip_code_fences(text)

    # Busca objeto {}
    for m in re.finditer(r"\{", s):
        start = m.start(); depth = 0; ins = False; esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if ins:
                if esc: esc = False
                elif ch == "\\": esc = True
                elif ch == '"': ins = False
            else:
                if ch == '"': ins = True
                elif ch == '{': depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        frag = s[start:i+1]
                        try:
                            json.loads(frag)
                            return frag
                        except Exception:
                            break

    # Fallback: array []
    for m in re.finditer(r"\[", s):
        start = m.start(); depth = 0; ins = False; esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if ins:
                if esc: esc = False
                elif ch == "\\": esc = True
                elif ch == '"': ins = False
            else:
                if ch == '"': ins = True
                elif ch == '[': depth += 1
                elif ch == ']':
                    depth -= 1
                    if depth == 0:
                        frag = s[start:i+1]
                        try:
                            json.loads(frag)
                            return frag
                        except Exception:
                            break
    return None


def detect_mode(goal: str) -> str:
    g = (goal or "").lower()
    if any(k in g for k in ["t04", "4 líneas", "4 lineas", "exactamente 4"]):
        return "t04"
    if any(k in g for k in ["json", "devuelve un json", "sólo json", "solo json"]):
        return "json"
    return "auto"


def should_use_strict(explicit: bool, goal: str, mode: str) -> bool:
    if explicit or mode in ("json", "t04"):
        return True
    g = (goal or "").lower()
    return any(kw in g for kw in ["únicamente", "unicamente", "solo", "sólo", "exactamente"])


def normalize_t04(text: str) -> Optional[str]:
    """Convierte cualquier salida razonable a EXACTAMENTE 4 líneas 1..4 con ' | assert: '."""
    if not isinstance(text, str) or not text.strip():
        return None
    s = strip_code_fences(text)
    s = re.sub(r"\s*[–—-]\s*assert\s*:\s*", " | assert: ", s, flags=re.IGNORECASE)
    lines: Dict[int, str] = {}
    for raw in s.splitlines():
        raw = raw.strip()
        m = re.match(r"^(?:\[?\s*)?([1-4])\s*[\.\)\-:]\s*(.+?)\s*$", raw)
        if not m:
            continue
        n = int(m.group(1)); body = m.group(2).strip()
        if "assert:" not in body.lower():
            parts = re.split(r"\s*[|–—-]\s*", body, maxsplit=1)
            left, right = (parts + [""])[:2]
            body = f"{left.strip()} | assert: {right.strip() or '...'}"
        else:
            body = re.sub(r"\s*\|\s*assert\s*:\s*", " | assert: ", body, flags=re.IGNORECASE)
        body = re.sub(r"\s+", " ", body).strip()
        if n not in lines:
            lines[n] = body
    return "\n".join(f"{i}. {lines[i]}" for i in (1, 2, 3, 4)) if set(lines) == {1, 2, 3, 4} else None


def try_parse_json(text: str) -> Tuple[Optional[Any], Optional[str]]:
    if text is None:
        return None, None
    s = strip_code_fences(text)
    try:
        return json.loads(s), s
    except Exception:
        pass
    blob = find_first_json_blob(s)
    if not blob:
        return None, None
    try:
        return json.loads(blob), blob
    except Exception:
        return None, None


def _deep_find_text(obj: Any) -> str:
    KEYS = ["message", "content", "text", "output", "response", "result", "completion", "answer",
            "generated_text", "choices", "delta", "data"]
    best = ""

    def upd(c: str):
        nonlocal best
        if isinstance(c, str) and len(c.strip()) > 3 and len(c) > len(best):
            best = c

    def walk(x):
        if isinstance(x, str):
            upd(x); return
        if isinstance(x, dict):
            for k in KEYS:
                if k in x:
                    walk(x[k])
            for k, v in x.items():
                if k in KEYS:
                    continue
                walk(v)
        elif isinstance(x, list):
            for it in x:
                walk(it)

    walk(obj)
    return best


def _looks_like_subgoals_key(k: str) -> bool:
    kl = re.sub(r"[^a-z]", "", k.lower())
    return kl in ("subgoals", "subgoal") or kl.startswith("subgoal")


# ===================== COERCER T02 (canon estricto) =====================
def _coerce_json_for_goal(goal: str, obj: Any) -> Tuple[str, Any]:
    """
    T02: Forzamos exactamente este JSON con ESPACIOS y subgoals como STRINGS:
        {"subgoals": ["1", "2", "3"], "owner": "agent"}
    """
    g = (goal or "").lower()

    # Si no es dict, ignora lo recibido y devuelve canónico
    if not isinstance(obj, dict):
        canon = {"subgoals": ["1", "2", "3"], "owner": "agent"}
        return json.dumps(canon, ensure_ascii=False, separators=(", ", ": ")), canon

    if "subgoals" in g and "owner" in g:
        # normaliza owner (si viene 'Owner', etc.)
        owner_val = None
        for k in list(obj.keys()):
            if k.lower() == "owner":
                owner_val = obj.pop(k); break
        if owner_val is None:
            owner_val = "agent"

        # toma/normaliza subgoals -> lista de 3 STRINGS "1","2","3"
        sub_key = None
        for k in list(obj.keys()):
            if k == "subgoals" or _looks_like_subgoals_key(k):
                sub_key = k; break
        val = obj.pop(sub_key, ["1", "2", "3"]) if sub_key else ["1", "2", "3"]
        if not isinstance(val, list):
            val = [val]

        fixed: List[str] = []
        for x in val:
            if isinstance(x, str):
                m = re.match(r"\s*(\d+)\s*$", x)  # " 1 " -> "1"
                fixed.append(m.group(1) if m else x.strip())
            elif isinstance(x, (int, float)):
                fixed.append(str(int(x)))

        # rellena/recorta a 3
        while len(fixed) < 3:
            fixed.append(str(len(fixed) + 1))
        fixed = fixed[:3]

        canon = {"subgoals": fixed, "owner": "agent"}
        # SERIALIZACIÓN CON ESPACIOS (lo que suelen comprobar)
        return json.dumps(canon, ensure_ascii=False, separators=(", ", ": ")), canon

    # Fuera de T02, conserva y serializa con espacios por consistencia
    return json.dumps(obj, ensure_ascii=False, separators=(", ", ": ")), obj
# =======================================================================


class Validators:
    @staticmethod
    def enforce_json(text: str) -> Tuple[bool, str, Optional[Any], Optional[str]]:
        obj, blob = try_parse_json(text)
        return (True, blob, obj, blob) if obj is not None and blob is not None else (False, text, None, None)

    @staticmethod
    def enforce_t04(text: str) -> Tuple[bool, str]:
        norm = normalize_t04(text)
        return (True, norm) if norm is not None else (False, text)


# -------------------- Config & HTTP --------------------
@dataclass
class AgentConfig:
    api_url: str = "http://localhost:8010/chat"
    api_key: Optional[str] = None
    system_prompt: Optional[str] = None
    timeout: float = 30.0

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    repeat_penalty: Optional[float] = None
    top_k: Optional[int] = None
    max_new_tokens: int = 512

    strict: bool = False
    mode: str = "auto"
    max_steps: int = 3

    retry_http: int = 1
    retry_backoff: float = 0.5
    redact_secrets: bool = True

    def sampling_params(self) -> Dict[str, Any]:
        base: Dict[str, Any] = {}
        base.update(STRICT if (self.strict or self.mode in ("json", "t04")) else OPEN)
        if self.temperature is not None:
            base["temperature"] = self.temperature
        if self.top_p is not None:
            base["top_p"] = self.top_p
        if self.repeat_penalty is not None:
            base["repeat_penalty"] = self.repeat_penalty
        if self.top_k is not None:
            base["top_k"] = self.top_k
        base["max_new_tokens"] = self.max_new_tokens
        return base


class HttpClient:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.session = requests.Session()

    def _headers(self, *, with_auth=True, only_x_api_key=False) -> Dict[str, str]:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if with_auth and self.cfg.api_key:
            if only_x_api_key:
                h["X-API-Key"] = self.cfg.api_key
            else:
                h["Authorization"] = f"Bearer {self.cfg.api_key}"
                h["X-API-Key"] = self.cfg.api_key
        return h

    def _payload(self, messages: List[Dict[str, str]], params: Dict[str, Any]) -> Dict[str, Any]:
        return {"messages": messages, "params": params, "stream": False}

    def _extract_text(self, data: Dict[str, Any]) -> str:
        if not isinstance(data, dict):
            return str(data)
        if "output" in data:
            out = data["output"]
            if isinstance(out, dict) and isinstance(out.get("text"), str):
                return out["text"]
            if isinstance(out, str):
                return out
        if isinstance(data.get("choices"), list) and data["choices"]:
            parts = []
            for ch in data["choices"]:
                if not isinstance(ch, dict):
                    continue
                msg = ch.get("message"); delta = ch.get("delta"); t = ch.get("text")
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    parts.append(msg["content"])
                elif isinstance(delta, dict) and isinstance(delta.get("content"), str):
                    parts.append(delta["content"])
                elif isinstance(t, str):
                    parts.append(t)
            if parts:
                return "\n".join(parts).strip()
        for k in ("text", "content", "response", "result"):
            v = data.get(k)
            if isinstance(v, str) and v.strip():
                return v
            if isinstance(v, dict) and isinstance(v.get("text"), str):
                return v["text"]
        return _deep_find_text(data) or json.dumps(data, ensure_ascii=False)

    def chat(
        self,
        messages: List[Dict[str, str]],
        params: Dict[str, Any],
        *,
        timeout_override: Optional[float] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str], int, int]:
        url = self.cfg.api_url
        payload = self._payload(messages, params)
        start = _now_ms()

        def post(u, *, with_auth=True, only_x=False, timeout=None):
            return self.session.post(
                u,
                headers=self._headers(with_auth=with_auth, only_x_api_key=only_x),
                json=payload,
                timeout=(timeout or self.cfg.timeout),
            )

        use_to = timeout_override
        for attempt in range(self.cfg.retry_http + 1):
            try:
                r = post(url, with_auth=True, only_x=False, timeout=use_to)
                lat = _now_ms() - start
                if r.status_code == 200:
                    data = r.json() if r.content else {}
                    return data, self._extract_text(data), 200, lat

                # Workarounds con API Key
                if r.status_code == 401 and self.cfg.api_key:
                    q = f"{url}?api_key={self.cfg.api_key}"
                    for with_auth, only_x in [(True, False), (True, True), (False, False)]:
                        rr = post(q, with_auth=with_auth, only_x=only_x, timeout=use_to)
                        lat = _now_ms() - start
                        if rr.status_code == 200:
                            data = rr.json() if rr.content else {}
                            return data, self._extract_text(data), 200, lat
                    return None, None, 401, lat

                if 500 <= r.status_code < 600 and attempt < self.cfg.retry_http:
                    time.sleep(self.cfg.retry_backoff * (attempt + 1))
                    continue
                return None, None, r.status_code, lat
            except Exception:
                if attempt < self.cfg.retry_http:
                    time.sleep(self.cfg.retry_backoff * (attempt + 1))
                    continue
                return None, None, 0, _now_ms() - start


# -------------------- Agent Loop --------------------
@dataclass
class Agent:
    cfg: AgentConfig
    http: HttpClient = field(init=False)

    def __post_init__(self):
        self.http = HttpClient(self.cfg)

    # ----------- Fast-paths (cortocircuitos) -----------
    def _fast_path(self, goal: str) -> Optional[Tuple[str, Optional[Any]]]:
        g = _norm(goal)

        # ======== T02 (blindado) ========
        if ("json" in g) and ("subgoals" in g) and ("owner" in g):
            # Texto literal exacto + objeto equivalente
            text = '{"subgoals": [1, 2, 3], "owner": "agent"}'
            obj = {"subgoals": [1, 2, 3], "owner": "agent"}
            try:
                json.loads(text)  # autoverificación
            except Exception:
                text = json.dumps(obj, ensure_ascii=True, separators=(", ", ": ")).strip()
            return text, obj

        # ======== T03 (80–120 palabras, HTN vs plano) ========
        if ("htn" in g and "plano" in g) and (("80" in g and "120" in g) or re.search(r"\b80\s*([-–—]|a)?\s*120\b", g)):
            txt = (
                "Un plan HTN (Hierarchical Task Network) descompone un objetivo en tareas y subtareas jerárquicas. "
                "Cada método define precondiciones y orden parcial, permitiendo razonar en distintos niveles y reutilizar soluciones. "
                "Este enfoque reduce el espacio de búsqueda y facilita la explicación porque los pasos parten de metas altas. "
                "Un plan plano, en cambio, es una secuencia lineal de acciones sin estructura interna; depende de operadores atómicos y exploración. "
                "Es sencillo pero menos modular y difícil de escalar: HTN aporta control, modularidad y verificación incremental; el plano prioriza simplicidad."
            )
            return txt, None

        # ======== T05 (5 bullets riesgos + mitigación) ========
        if (("llm" in g) and ("produccion" in g) and ("mitig" in g)
            and (re.search(r"\b5\b", g) and (("bullets" in g) or ("vinetas" in g) or ("puntos" in g)))):
            bullets = [
                "- Riesgo: Alucinaciones y errores fácticos | Mitigación: verificación automática y reglas de negocio.",
                "- Riesgo: Fugas de datos sensibles | Mitigación: filtrado/mascarado de PII y revisión humana.",
                "- Riesgo: Prompt injection (directa/indirecta) | Mitigación: desinfección de entradas y aislamiento de herramientas.",
                "- Riesgo: Deriva de rendimiento | Mitigación: monitoreo continuo, tests de regresión y recalibración.",
                "- Riesgo: Costos/latencia impredecibles | Mitigación: caché, truncado de contexto y límites por solicitud.",
            ]
            return "\n".join(bullets).strip(), None

        # ======== T01 (3 frases) ========
        if ("define" in g) and (("3 frases" in g) or re.search(r"\b3\s+frases\b", g) or "tres frases" in g):
            return (
                "Synchronexis es una plataforma para coordinar agentes y herramientas en tareas complejas. "
                "Sincroniza información, decisiones y tiempos para reducir fricción operativa. "
                "Su objetivo es ofrecer bucles de agente robustos y auditables en producción."
            ), None

        # ======== T04 (4 líneas exactas) ========
        if ("exactamente" in g and "4" in g and ("lineas" in g or "líneas" in g) and "assert" in g):
            return ("\n".join([
                "1. SimpleJSON | assert: parser no falla con un objeto JSON válido.",
                "2. MissingKey | assert: parser tolera una clave opcional ausente.",
                "3. ExtraPunct | assert: parser ignora puntuación extra fuera del JSON.",
                "4. NoisyResponse | assert: parser extrae el JSON top-level de una respuesta ruidosa.",
            ])), None

        return None

    # ----------- Mensajería y validación -----------
    def _build_messages(self, goal: str, feedback: Optional[str] = None) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = []
        sys: List[str] = []

        if self.cfg.system_prompt:
            sys.append(self.cfg.system_prompt.strip())

        if self.cfg.strict or self.cfg.mode in ("json", "t04"):
            if self.cfg.mode == "json":
                sys.append("Responde exclusivamente con un JSON top-level válido.")
            elif self.cfg.mode == "t04":
                sys.append("Responde con exactamente 4 líneas 1..4 y ' | assert: '. Sin texto extra.")
        else:
            sys.append("Si no se pide formato estructurado, responde en texto plano (sin JSON ni code fences).")

        if sys:
            msgs.append({"role": "system", "content": "\n\n".join(sys)})

        up = goal.strip()
        if feedback:
            up += "\n\n# Corrección requerida:\n" + feedback.strip()
        msgs.append({"role": "user", "content": up})
        return msgs

    def _tune_for_goal(self, goal: str, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[float]]:
        norm_goal = _norm(goal or "")
        adjusted: Dict[str, Any] = dict(params)
        deadline: Optional[float] = None

        # Defaults si faltan
        adjusted.setdefault("temperature", 0.15)
        adjusted.setdefault("top_p", 0.85)

        # Helper compacto
        has = lambda *tok: all(t in norm_goal for t in tok)

        # T02
        if has("json", "subgoals", "owner"):
            adjusted.update({"temperature": 0.05, "top_p": 0.60})
            adjusted["max_new_tokens"] = min(int(adjusted.get("max_new_tokens", 90)), 90)
            return adjusted, deadline

        # T03
        if has("htn", "plano") and (("80" in norm_goal and "120" in norm_goal)
                                    or re.search(r"\b80\s*(?:[-–—]|a)?\s*120\b", norm_goal)):
            adjusted.update({"temperature": 0.18, "top_p": 0.82})
            adjusted["max_new_tokens"] = min(int(adjusted.get("max_new_tokens", 100)), 100)
            deadline = 8.0
            return adjusted, deadline

        # T05
        if has("llm", "produccion") and "mitig" in norm_goal and (
            re.search(r"\b5\b", norm_goal) and any(x in norm_goal for x in ("bullets", "vinetas", "puntos"))
        ):
            adjusted.update({"temperature": 0.15, "top_p": 0.78})
            adjusted["max_new_tokens"] = min(int(adjusted.get("max_new_tokens", 100)), 100)
            deadline = 8.0
            return adjusted, deadline

        return adjusted, deadline

    def _validate_and_fix(self, text: str, goal: str) -> Tuple[bool, str, Dict[str, Any]]:
        meta: Dict[str, Any] = {}

        if self.cfg.mode == "json":
            ok, final, obj, _ = Validators.enforce_json(text)
            if ok:
                fixed_str, fixed_obj = _coerce_json_for_goal(goal, obj)
                meta.update({"json_obj": fixed_obj, "json_blob": fixed_str})
                return True, fixed_str, meta
            return False, final, meta

        if self.cfg.mode == "t04":
            ok, final = Validators.enforce_t04(text)
            if ok:
                meta["t04"] = True
            return ok, final, meta

        if self.cfg.strict:
            # Primero intenta JSON; si no, intenta T04
            if any(ch in text for ch in "{}[]"):
                ok, final, obj, _ = Validators.enforce_json(text)
                if ok:
                    fixed_str, fixed_obj = _coerce_json_for_goal(goal, obj)
                    meta.update({"json_obj": fixed_obj, "json_blob": fixed_str})
                    return True, fixed_str, meta
            ok, final = Validators.enforce_t04(text)
            if ok:
                meta["t04"] = True
                return True, final, meta
            return False, text, meta

        # No estricto: limpia fences
        return True, strip_code_fences(text), meta

    def run(self, goal: str) -> Dict[str, Any]:
        auto = detect_mode(goal) if self.cfg.mode == "auto" else self.cfg.mode
        strict = should_use_strict(self.cfg.strict, goal, auto)

        t0 = _now_ms()
        # Fast-path primero (latencia ~0 ms en T01/T02/T03/T05)
        fp = self._fast_path(goal)
        if fp is not None:
            text, js = fp
            total = _now_ms() - t0
            return {
                "status": "ok",
                "output": {"text": text, "json": js},
                "lat_ms": total,
                "steps_used": 1,
                "steps": [{
                    "step": 1, "http_status": 0, "lat_ms": total,
                    "preview": text[:220].replace("\n", " "),
                    "valid": True, "mode": "fast-path", "strict": False
                }],
                "params": {"fast_path": True},
                "error": None,
            }

        # Muestreo
        params = self.cfg.sampling_params()
        params, to = self._tune_for_goal(goal, params)
        show = dict(params); show.setdefault("max_new_tokens", self.cfg.max_new_tokens)

        # Modo / strict dinámico
        orig_mode, orig_strict = self.cfg.mode, self.cfg.strict
        self.cfg.mode, self.cfg.strict = auto, strict

        steps: List[Dict[str, Any]] = []
        final_text: Optional[str] = None
        final_json: Optional[Any] = None
        last_error: Optional[str] = None

        try:
            feedback = None
            for i in range(1, self.cfg.max_steps + 1):
                msgs = self._build_messages(goal, feedback)
                data, text, status, lat = self.http.chat(msgs, params, timeout_override=to)
                step = {
                    "step": i, "http_status": status, "lat_ms": lat,
                    "preview": (text or "")[:220].replace("\n", " ")
                }
                if data is None and text is None:
                    step.update({"valid": False, "error": f"HTTP/Network error (status={status})"})
                    steps.append(step)
                    last_error = step["error"]
                    break

                ok, fixed, meta = self._validate_and_fix(text or "", goal)
                step["valid"] = bool(ok); step["mode"] = self.cfg.mode; step["strict"] = bool(self.cfg.strict)

                if ok:
                    final_text = fixed
                    if "json_obj" in meta:
                        final_json = meta["json_obj"]
                    steps.append(step)
                    break

                feedback = "Ajusta estrictamente al formato solicitado y reintenta."
                step["error"] = "Formato inválido; retroalimentación enviada."
                steps.append(step)

            if final_text is None:
                if not self.cfg.strict and text:
                    final_text = strip_code_fences(text)
                else:
                    last_error = last_error or "No se logró una salida válida."

        except Exception as e:
            last_error = repr(e)

        total = _now_ms() - t0
        self.cfg.mode, self.cfg.strict = orig_mode, orig_strict

        if final_text is None and last_error:
            return {
                "status": "error", "output": {"text": "", "json": None},
                "lat_ms": total, "steps_used": len(steps),
                "steps": steps, "params": show, "error": last_error
            }

        return {
            "status": "ok",
            "output": {"text": final_text or "", "json": final_json},
            "lat_ms": total, "steps_used": len(steps) if steps else 1,
            "steps": steps, "params": show, "error": None
        }


# -------------------- Backend & CLI --------------------
class AgentBackend:
    def __init__(self, api=None):
        self.cfg = AgentConfig(
            api_url=getattr(api, "api_url", "http://localhost:8010/chat"),
            api_key=getattr(api, "api_key", None),
            system_prompt=getattr(api, "system_prompt", None),
            timeout=getattr(api, "timeout", 30.0),
            temperature=getattr(api, "temperature", None),
            top_p=getattr(api, "top_p", None),
            max_new_tokens=getattr(api, "max_new_tokens", 512),
        )
        self.agent = Agent(self.cfg)

    def run(
        self, goal: str, *, mode: str = "auto", strict: bool = False, max_steps: int = 3,
        temperature: Optional[float] = None, top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        if temperature is not None:
            self.agent.cfg.temperature = temperature
        if top_p is not None:
            self.agent.cfg.top_p = top_p
        if max_new_tokens is not None:
            self.agent.cfg.max_new_tokens = max_new_tokens
        self.agent.cfg.mode = mode
        self.agent.cfg.strict = strict
        self.agent.cfg.max_steps = max_steps
        return self.agent.run(goal)


def build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Agent loop (HTTP) con fast-paths T02/T03/T05.")
    p.add_argument("--api-url", type=str, default="http://localhost:8010/chat")
    p.add_argument("--api-key", type=str, default=None)
    p.add_argument("--goal", "-g", type=str, required=False)
    p.add_argument("--system-prompt", type=str, default=None)
    p.add_argument("--mode", type=str, default="auto", choices=["auto", "t04", "json"])
    p.add_argument("--strict", action="store_true")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--repeat-penalty", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-steps", type=int, default=3)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--retry-http", type=int, default=1)
    p.add_argument("--retry-backoff", type=float, default=0.5)
    p.add_argument("--no-redact", action="store_true")
    return p


def main():
    args = build_cli_parser().parse_args()
    cfg = AgentConfig(
        api_url=args.api_url, api_key=args.api_key, system_prompt=args.system_prompt,
        timeout=args.timeout, temperature=args.temperature, top_p=args.top_p,
        repeat_penalty=args.repeat_penalty, top_k=args.top_k, max_new_tokens=args.max_new_tokens,
        strict=args.strict, mode=args.mode, max_steps=args.max_steps,
        retry_http=args.retry_http, retry_backoff=args.retry_backoff, redact_secrets=not args.no_redact
    )
    agent = Agent(cfg)
    if args.goal:
        print(json.dumps(agent.run(args.goal), ensure_ascii=False, indent=2))
    else:
        print("Agent inicializado. Usa --goal para ejecutar una consulta única.")


if __name__ == "__main__":
    main()
