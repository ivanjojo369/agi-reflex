# AGI RefleX – AGI local modular y auto-reflexiva

Este proyecto busca construir una AGI local, autosuficiente y capaz de razonar de manera multimodal, con:
- Memoria contextual semántica persistente
- Planificación de tareas activa
- Capacidad reflexiva y adaptable a múltiples dominios

La arquitectura permite usar modelos GGUF locales como OpenChat 3.5, y está preparada para integrarse fácilmente con modelos futuros como GPT-4.5, Claude, o Gemini.

## Objetivos
- Ejecutar una AGI local sin conexión a internet
- Reemplazar modelos fácilmente como si fueran módulos enchufables
- Mantener memoria contextual precisa entre sesiones
- Implementar un planner con metas jerárquicas

## Estado del proyecto
🔨 En desarrollo (Julio 2025). Requiere GPU y/o entorno CPU optimizado.

## Estructura
- `src/agi_interface.py`: núcleo conversacional
- `src/model_interface.py`: interfaz de modelos LLM plug & play
- `src/planner.py`: gestión de metas y tareas
- `src/memory.py`: sistema de memoria semántica en JSON

## Instalación
```bash
pip install -r requirements.txt
