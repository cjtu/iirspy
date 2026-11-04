"""Base class of the level configs (`L1Config`, `L2Config`): named presets, overrides and a JSON round trip.

A config is a frozen dataclass whose fields each name one method choice. `resolve` accepts a preset name or an
instance plus field overrides; `to_attrs` is what lands in a product's `l1_config` / `l2_config` attr.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar, Self


class Config:
    """Preset lookup and JSON round trip shared by the level configs."""

    PRESETS: ClassVar[dict[str, dict]] = {}

    @classmethod
    def preset(cls, name: str, **overrides) -> Self:
        """The named preset with `overrides` applied. Raises on an unknown preset or field."""
        if name not in cls.PRESETS:
            raise ValueError(f"unknown {cls.__name__} preset {name!r}; choose from {tuple(cls.PRESETS)}")
        return cls(**{**cls.PRESETS[name], **overrides})

    @classmethod
    def resolve(cls, config="iirspy", **overrides) -> Self:
        """A preset name or an instance, with `overrides` applied."""
        if isinstance(config, cls):
            return dataclasses.replace(config, **overrides) if overrides else config  # type: ignore[type-var]
        return cls.preset(config, **overrides)

    def to_attrs(self) -> dict:
        """Every field as JSON-safe values; `type(cfg)(**cfg.to_attrs())` rebuilds the config."""
        out = {}
        for f in dataclasses.fields(self):  # type: ignore[arg-type]
            v = getattr(self, f.name)
            if callable(v):
                v = getattr(v, "__name__", "custom")
            out[f.name] = [list(e) if isinstance(e, tuple) else e for e in v] if isinstance(v, tuple) else v
        return out
