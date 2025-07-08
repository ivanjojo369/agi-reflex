from llama_cpp_adapter import LlamaCppAdapter
from memory import ContextMemory
from planner import TaskPlanner
from agi_core import AGIInterface
import os
import sys

def check_model_exists(path):
    if not os.path.exists(path):
        print(f"\n🛑 ERROR: El modelo no fue encontrado en: {path}")
        print("📁 Verifica que el archivo .gguf esté en la carpeta 'models/'")
        sys.exit(1)

# Configuración de modelo
model_path = "models/openchat-3.5-1210.Q4_K_M.gguf"
check_model_exists(model_path)

model = LlamaCppAdapter(model_path=model_path, n_ctx=4096, n_threads=8, n_batch=64)
model.load_model()

# Carga módulos
memory = ContextMemory()
planner = TaskPlanner()
agi = AGIInterface(model_adapter=model, memory=memory, planner=planner)

# Ejecuta la AGI
agi.start()
