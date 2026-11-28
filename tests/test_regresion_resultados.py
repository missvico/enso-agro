"""Tests de regresión contra JSONs guardados (usando ENSO_N_MC=20 para ejecución rápida)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

RAIZ = Path(__file__).resolve().parents[1]
PROC = RAIZ / "data" / "processed"
RES = RAIZ / "results"
sys.path.insert(0, str(RAIZ / "src"))

from utils_ts import anomalias, oni_campanias, rindes
from wavelet import fase_circular_media, grilla_morlet, umbral_montecarlo_ar1, wavelet_coherence


def test_regresion_h1_betas_contra_etapa2_json():
    etapa2 = json.loads((RES / "etapa2.json").read_text(encoding="utf-8"))
    rend = pd.read_parquet(PROC / "rendimientos_anuales.parquet")
    camp = pd.read_parquet(PROC / "campanias.parquet")

    A = anomalias(rindes(rend), metodo="lowess")
    O = oni_campanias(camp)

    for cultivo, col_oni in (("soja", "oni_JAS_0"), ("maiz", "oni_JAS_0"), ("trigo", "oni_JAS_0")):
        d = pd.DataFrame({
            "y": A[cultivo],
            "y1": A[cultivo].shift(1),
            "oni": O.loc[1970:2024, col_oni],
        }).dropna()
        mod = smf.ols("y ~ y1 + oni", d).fit()
        esp = etapa2["h1"][cultivo]
        assert float(mod.params["oni"]) == pytest.approx(esp["beta_%/°C"], abs=1e-4)
        assert float(mod.pvalues["oni"]) == pytest.approx(esp["p_F"], abs=1e-5)


def test_regresion_m3_skill_soja_y_maiz_contra_etapa3b_json():
    etapa3b = json.loads((RES / "etapa3b_rindes.json").read_text(encoding="utf-8"))
    tabla_esp = {(r["cultivo"], r["modelo"]): r for r in etapa3b["tabla"]}

    for cultivo in ("soja", "maiz"):
        df = pd.read_csv(RES / f"pronosticos_rolling_{cultivo}.csv").set_index("campania")
        e2 = df["obs"] - df["M2 ARIMA"]
        e3 = df["obs"] - df["M3 ARIMAX (ONI jul–sep)"]
        rmse_m3 = float(np.sqrt(np.mean(e3 ** 2)))
        skill_m3 = float(100.0 * (1.0 - np.mean(e3 ** 2) / np.mean(e2 ** 2)))

        esp = tabla_esp[(cultivo, "M3 ARIMAX (ONI jul–sep)")]
        assert rmse_m3 == pytest.approx(esp["RMSE kg/ha"], abs=0.05)
        assert skill_m3 == pytest.approx(esp["skill vs ARIMA %"], abs=0.05)


def test_regresion_wavelet_rapido_con_enso_n_mc_20(monkeypatch):
    monkeypatch.setenv("ENSO_N_MC", "20")
    ext = json.loads((RES / "etapa4_extensiones.json").read_text(encoding="utf-8"))
    bw_esp = ext["bloque5_coherencia_wavelet"]

    enso = pd.read_parquet(PROC / "enso_mensual.parquet")
    clima = pd.read_parquet(PROC / "clima_pampeano_mensual.parquet")
    m_wc = pd.concat([enso["n34_anom"], clima["precip_z"]], axis=1).loc["1981":"2025"].dropna()
    xw = (m_wc["n34_anom"].to_numpy(float) - m_wc["n34_anom"].mean()) / m_wc["n34_anom"].std()
    yw = (m_wc["precip_z"].to_numpy(float) - m_wc["precip_z"].mean()) / m_wc["precip_z"].std()

    grilla = grilla_morlet(n=len(xw))
    R2_obs, phase_obs, _ = wavelet_coherence(xw, yw, grilla=grilla)
    assert float(R2_obs.mean()) == pytest.approx(bw_esp["R2_global_mean"], abs=1e-6)
    assert float(R2_obs.max()) == pytest.approx(bw_esp["R2_global_max"], abs=1e-6)

    sig95_scale, n_mc = umbral_montecarlo_ar1(
        float(m_wc["n34_anom"].autocorr(1)),
        float(m_wc["precip_z"].autocorr(1)),
        grilla=grilla,
    )
    assert n_mc == 20
    sig_banda = (R2_obs / sig95_scale[:, None] >= 1.0) & grilla["inside_coi"] & (
        (grilla["periods"] >= 2.0) & (grilla["periods"] <= 7.0)
    )[:, None]
    fase_stat = fase_circular_media(phase_obs, sig_banda)
    assert abs(fase_stat["fase_media_grados"]) < 35.0
    assert fase_stat["longitud_resultante_R"] > 0.6
