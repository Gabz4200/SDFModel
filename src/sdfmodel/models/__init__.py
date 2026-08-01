from collections.abc import Callable

from sdfmodel.models.base import BaseModel
from sdfmodel.models.cross_attn_sdf import CrossAttnSDFModel
from sdfmodel.models.sdf_mlp import SDFMLP
from sdfmodel.models.vector_sdf import VectorSDFModel

_MODEL_REGISTRY: dict[str, type[BaseModel]] = {}


def register_model(name: str) -> Callable[[type[BaseModel]], type[BaseModel]]:
    def decorator(cls: type[BaseModel]) -> type[BaseModel]:
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


register_model("sdf_mlp")(SDFMLP)
register_model("cross_attn_sdf")(CrossAttnSDFModel)
register_model("vector_sdf")(VectorSDFModel)


def list_models() -> list[str]:
    return sorted(_MODEL_REGISTRY.keys())


def build_model(name: str, **kwargs) -> BaseModel:
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model name '{name}'. Registered: {list_models()}")
    return _MODEL_REGISTRY[name](**kwargs)


__all__ = [
    "SDFMLP",
    "BaseModel",
    "CrossAttnSDFModel",
    "VectorSDFModel",
    "build_model",
    "list_models",
    "register_model",
]
