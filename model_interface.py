# model_interface.py

from abc import ABC, abstractmethod

class BaseModelInterface(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, context: list[str] = None) -> str:
        """Genera una respuesta textual basada en el prompt y contexto opcional."""
        pass

    @abstractmethod
    def load_model(self):
        """Carga el modelo en memoria (si aplica)."""
        pass
