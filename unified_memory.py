import os
import json
import faiss
import numpy as np
import logging
import threading

class UnifiedMemory:
    def __init__(self, memory_dir="memory_store", vector_dim=1536, use_gpu=False):
        self.logger = logging.getLogger("UnifiedMemory")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.memory_dir = memory_dir
        self.vector_dim = vector_dim
        self.use_gpu = use_gpu
        self.lock = threading.Lock()

        os.makedirs(self.memory_dir, exist_ok=True)

        self.index_file = os.path.join(self.memory_dir, "faiss_index.bin")
        self.vector_file = os.path.join(self.memory_dir, "vector_memories.json")
        self.memories = []
        self.index = None

        self._load_memories()
        self._init_faiss_index()

    # -----------------------------
    # Inicialización de FAISS
    # -----------------------------
    def _init_faiss_index(self):
        try:
            if os.path.exists(self.index_file):
                self.index = faiss.read_index(self.index_file)
                self.logger.info("[UM] Índice FAISS cargado correctamente.")
            else:
                self.index = faiss.IndexFlatL2(self.vector_dim)
                self.logger.info("[UM] Nuevo índice FAISS creado.")

            if self.use_gpu:
                try:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                    self.logger.info("[UM] FAISS usando GPU.")
                except Exception as e:
                    self.logger.warning(f"[UM] No se pudo usar GPU para FAISS: {e}")
        except Exception as e:
            self.logger.error(f"[UM] Error inicializando FAISS: {e}")
            self.index = faiss.IndexFlatL2(self.vector_dim)

    # -----------------------------
    # Cargar memorias desde disco
    # -----------------------------
    def _load_memories(self):
        try:
            if os.path.exists(self.vector_file):
                with open(self.vector_file, "r", encoding="utf-8") as f:
                    self.memories = json.load(f)
                self.logger.info(f"[UM] {len(self.memories)} memorias cargadas desde disco.")
            else:
                self.memories = []
        except Exception as e:
            self.logger.error(f"[UM] Error cargando memorias: {e}")
            self.memories = []

    # -----------------------------
    # Guardar memorias y FAISS
    # -----------------------------
    def save_to_disk(self):
        try:
            with open(self.vector_file, "w", encoding="utf-8") as f:
                json.dump(self.memories, f, ensure_ascii=False, indent=2)
            faiss.write_index(self.index, self.index_file)
            self.logger.info("[UM] Memorias y FAISS guardados en disco.")
        except Exception as e:
            self.logger.error(f"[UM] Error guardando memorias: {e}")

    # -----------------------------
    # Agregar memoria
    # -----------------------------
    def add_memory(self, text, metadata=None):
        try:
            if not isinstance(text, str):
                text = str(text)

            embedding = self.embed_text(text)
            with self.lock:
                self.index.add(np.array([embedding], dtype=np.float32))
                self.memories.append({
                    "text": text,
                    "metadata": metadata or {}
                })
            self.save_to_disk()
            self.logger.info(f"[UM] Memoria agregada: {text[:50]}")
        except Exception as e:
            self.logger.error(f"[UM] Error agregando memoria: {e}")

    # -----------------------------
    # Buscar memorias similares
    # -----------------------------
    def search_memory(self, query_vector, top_k=5):
        """
        Busca recuerdos similares. Si recibe texto, genera embedding automáticamente.
        """
        try:
            if isinstance(query_vector, dict):
                query_vector = query_vector.get("input", "")
            if isinstance(query_vector, str):
                query_vector = self.embed_text(query_vector)

            query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
            distances, indices = self.index.search(query_vector, top_k)

            resultados = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx != -1 and idx < len(self.memories):
                    resultados.append({
                        "text": self.memories[idx]["text"],
                        "metadata": self.memories[idx].get("metadata", {}),
                        "distance": float(dist)
                    })

            return resultados

        except Exception as e:
            self.logger.error(f"[UM] Error en search_memory: {e}")
            return []

    # -----------------------------
    # Generar embedding
    # -----------------------------
    def embed_text(self, text: str):
        """
        Genera un embedding para el texto dado.
        ⚠️ Reemplazar con tu modelo real (OpenChat u otro).
        """
        return np.random.rand(self.vector_dim).astype(np.float32)

    # -----------------------------
    # Utilidades
    # -----------------------------
    def get_memory_count(self):
        return len(self.memories)

    def export_memories(self):
        return self.memories
