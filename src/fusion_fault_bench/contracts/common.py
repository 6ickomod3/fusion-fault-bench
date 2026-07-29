"""Shared strict contract behavior."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ContractModel(BaseModel):
    """Immutable, strict, forward-incompatible-by-default contract base."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        serialize_by_alias=True,
        strict=True,
    )
