"""Simulation artifacts. Request Chrome Trace is the only default-wired one."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RequestProfileOutput:
    """Request-level Chrome Trace (existing writer subprocess)."""

    enabled: bool = False
    #: Explicit profile JSON path (overrides ``dir``).
    path: Optional[Path] = None
    #: Directory for ``request_profile.json`` when path is unset.
    dir: Optional[Path] = None


@dataclass
class ArtifactOutput:
    """Optional file artifact (metrics / requests / config snapshot)."""

    enabled: bool = False
    path: Optional[Path] = None


@dataclass
class OutputConfig:
    """Enable simulation outputs. All optional artifacts default off."""

    #: Shared directory for metrics / requests / config when per-file path is unset.
    dir: Optional[Path] = None
    request_profile: RequestProfileOutput = field(default_factory=RequestProfileOutput)
    metrics: ArtifactOutput = field(default_factory=ArtifactOutput)
    requests: ArtifactOutput = field(default_factory=ArtifactOutput)
    config_snapshot: ArtifactOutput = field(default_factory=ArtifactOutput)
