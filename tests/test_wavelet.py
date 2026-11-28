"""Tests unitarios sobre casos analíticos conocidos para src/wavelet.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from wavelet import (
    fase_circular_media,
    grilla_morlet,
    simular_ar1,
    umbral_montecarlo_ar1,
    wavelet_coherence,
)


def test_autocoherencia_es_uno():
    n = 180
    t = np.arange(n) / 12.0
    x = np.sin(2.0 * np.pi * t / 3.5) + 0.4 * np.cos(2.0 * np.pi * t / 1.5)
    R2, phase, _ = wavelet_coherence(x, x)
    assert np.allclose(R2, 1.0, atol=1e-10)
    assert np.allclose(phase, 0.0, atol=1e-10)


def test_dos_ruidos_independientes_dan_coherencia_baja():
    n = 480
    rng = np.random.default_rng(123)
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    R2, _, grilla = wavelet_coherence(x, y)
    mask_coi = grilla["inside_coi"] & (grilla["periods"][:, None] >= 1.0)
    assert float(R2[mask_coi].mean()) < 0.35


def test_seno_comun_4_anios_alta_coherencia():
    n = 480  # 40 años mensuales
    dt = 1.0 / 12.0
    t = np.arange(n) * dt
    rng = np.random.default_rng(7)
    x = np.sin(2.0 * np.pi * t / 4.0) + 0.05 * rng.standard_normal(n)
    y = 1.5 * np.sin(2.0 * np.pi * t / 4.0) + 0.05 * rng.standard_normal(n)

    R2, phase, grilla = wavelet_coherence(x, y, dt=dt)
    periods = grilla["periods"]
    idx_4 = int(np.argmin(np.abs(periods - 4.0)))
    mask_t = grilla["inside_coi"][idx_4]
    assert mask_t.any()
    assert float(R2[idx_4, mask_t].mean()) > 0.95

    fase_stat = fase_circular_media(phase, (np.arange(len(periods))[:, None] == idx_4) & grilla["inside_coi"])
    assert abs(fase_stat["fase_media_rad"]) < 0.05
    assert fase_stat["longitud_resultante_R"] > 0.99


def test_seno_desplazado_recupera_fase_esperada():
    n = 480
    dt = 1.0 / 12.0
    t = np.arange(n) * dt
    delta = np.pi / 3.0  # y retrasada pi/3 rad (x adelanta a y por +pi/3)
    x = np.cos(2.0 * np.pi * t / 4.0)
    y = np.cos(2.0 * np.pi * t / 4.0 - delta)

    R2, phase, grilla = wavelet_coherence(x, y, dt=dt)
    periods = grilla["periods"]
    idx_4 = int(np.argmin(np.abs(periods - 4.0)))
    mask = (np.arange(len(periods))[:, None] == idx_4) & grilla["inside_coi"]

    assert float(R2[mask].mean()) > 0.99
    fase_stat = fase_circular_media(phase, mask)
    assert np.isclose(fase_stat["fase_media_rad"], delta, atol=0.05)
    assert np.isclose(fase_stat["fase_media_grados"], 60.0, atol=3.0)
    assert fase_stat["longitud_resultante_R"] > 0.99


def test_umbral_montecarlo_respeta_variable_de_entorno(monkeypatch):
    monkeypatch.setenv("ENSO_N_MC", "10")
    grilla = grilla_morlet(n=240)
    sig95, n_mc_usado = umbral_montecarlo_ar1(0.85, 0.15, grilla=grilla, borde=30, seed=0)
    assert n_mc_usado == 10
    assert sig95.shape == (len(grilla["scales"]),)
    assert np.all((sig95 > 0.0) & (sig95 <= 1.0))

    rng = np.random.default_rng(1)
    ar = simular_ar1(0.7, b_size=3, n=120, rng=rng)
    assert ar.shape == (3, 120)
    assert np.allclose(ar.mean(axis=1), 0.0, atol=1e-12)
