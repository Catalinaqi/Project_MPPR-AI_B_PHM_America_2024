# src/phm_america_2024/registry/__init__.py
from __future__ import annotations

# Importamos el archivo de manera global para activar los decoradores
from phm_america_2024.registry import phase2_generator_registry

# Exponemos las funciones del despachador de manera limpia
from phm_america_2024.registry.generator_registry_registry import (
    register_artifact,
    write_output_artifacts,
)

__all__ = [
    "register_artifact",
    "write_output_artifacts",
]