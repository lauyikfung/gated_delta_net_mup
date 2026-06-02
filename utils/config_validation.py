from __future__ import annotations

from types import EllipsisType
from typing import Any, Mapping, cast

from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt, StrictStr, create_model


def _to_pydantic_type(py_type: type[Any]) -> Any:
    if py_type is bool:
        return StrictBool
    if py_type is int:
        return StrictInt
    if py_type is str:
        return StrictStr
    if py_type is float:
        # Int -> float coercion is convenient in configs (e.g. dropout=0).
        return float
    return py_type


def validate_config(
    values: Mapping[str, Any],
    expected_types: Mapping[str, type[Any]],
    *,
    model_name: str = "Config",
) -> dict[str, Any]:
    fields: dict[str, tuple[Any, EllipsisType]] = {
        key: (_to_pydantic_type(py_type), ...) for key, py_type in expected_types.items()
    }
    model: type[BaseModel] = create_model(
        model_name,
        __config__=ConfigDict(extra="forbid"),
        **cast(Any, fields),
    )
    validated = model.model_validate(dict(values))
    return validated.model_dump()
