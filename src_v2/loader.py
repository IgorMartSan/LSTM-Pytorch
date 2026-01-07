from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType


def load_module(base_file: str, filename: str, name: str) -> ModuleType:
    path = Path(base_file).parent / filename
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Falha ao carregar modulo: {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
