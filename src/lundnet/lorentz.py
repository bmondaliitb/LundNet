"""Minimal Lorentz vector helpers.

This project historically relied on `uproot_methods` / `uproot3_methods` for
`TLorentzVector` and `TLorentzVectorArray`.  Those packages are not compatible
with Python 3.12+ (they use deprecated import machinery).

Only a small subset of the API is required by LundNet's dataset code:
- construction from cartesian components
- properties: pt, eta, energy
- methods: delta_phi(other)
- array sum()

This module provides that subset using NumPy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence, Union

import numpy as np


def _wrap_delta_phi(dphi: np.ndarray) -> np.ndarray:
    """Wrap delta-phi into (-pi, pi]."""
    return (dphi + math.pi) % (2.0 * math.pi) - math.pi


def _eta(px: np.ndarray, py: np.ndarray, pz: np.ndarray) -> np.ndarray:
    p = np.sqrt(px * px + py * py + pz * pz)
    # Avoid division by zero / log of non-positive.
    denom = np.maximum(p - pz, 1e-12)
    numer = np.maximum(p + pz, 1e-12)
    return 0.5 * np.log(numer / denom)


@dataclass(frozen=True)
class TLorentzVector:
    px: float
    py: float
    pz: float
    E: float

    @property
    def pt(self) -> float:
        return float(math.hypot(self.px, self.py))

    @property
    def energy(self) -> float:
        return float(self.E)

    @property
    def phi(self) -> float:
        return float(math.atan2(self.py, self.px))

    @property
    def eta(self) -> float:
        px = np.asarray(self.px, dtype=np.float64)
        py = np.asarray(self.py, dtype=np.float64)
        pz = np.asarray(self.pz, dtype=np.float64)
        return float(_eta(px, py, pz))

    def delta_phi(self, other: "TLorentzVector") -> float:
        dphi = np.asarray(self.phi - other.phi, dtype=np.float64)
        return float(_wrap_delta_phi(dphi))


class TLorentzVectorArray:
    """Array of Lorentz vectors (NumPy-backed)."""

    def __init__(self, px: np.ndarray, py: np.ndarray, pz: np.ndarray, E: np.ndarray):
        self.px = np.asarray(px)
        self.py = np.asarray(py)
        self.pz = np.asarray(pz)
        self.E = np.asarray(E)

    @classmethod
    def from_cartesian(
        cls,
        px: Sequence[float],
        py: Sequence[float],
        pz: Sequence[float],
        E: Sequence[float],
    ) -> "TLorentzVectorArray":
        return cls(np.asarray(px, dtype=np.float64), np.asarray(py, dtype=np.float64), np.asarray(pz, dtype=np.float64), np.asarray(E, dtype=np.float64))

    @property
    def pt(self) -> np.ndarray:
        return np.sqrt(self.px * self.px + self.py * self.py)

    @property
    def energy(self) -> np.ndarray:
        return self.E

    @property
    def phi(self) -> np.ndarray:
        return np.arctan2(self.py, self.px)

    @property
    def eta(self) -> np.ndarray:
        return _eta(self.px, self.py, self.pz)

    def sum(self) -> TLorentzVector:
        return TLorentzVector(float(self.px.sum()), float(self.py.sum()), float(self.pz.sum()), float(self.E.sum()))

    def delta_phi(self, other: TLorentzVector) -> np.ndarray:
        dphi = self.phi - other.phi
        return _wrap_delta_phi(dphi)
