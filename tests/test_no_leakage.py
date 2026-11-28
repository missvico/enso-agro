"""Tests anti-leakage temporal sobre rolling_extensiones_h1, rolling_arimax_h2 y puente Niño 3.4."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
PROC = RAIZ / "data" / "processed"
RES = RAIZ / "results"
sys.path.insert(0, str(RAIZ / "src"))

from rolling import (
    pronosticar_puente_n34_djf,
    rolling_arimax_h2,
    rolling_extensiones_h1,
    rolling_puente_rindes,
)
from utils_ts import rindes


def _cargar_tablas():
    enso = pd.read_parquet(PROC / "enso_mensual.parquet")
    rend = pd.read_parquet(PROC / "rendimientos_anuales.parquet")
    Y = rindes(rend)
    oni_mes = lambda mes, desfase=0: enso["oni"][enso.index.month == mes].set_axis(
        enso.index[enso.index.month == mes].year - desfase
    )
    X = pd.DataFrame({
        "oni_jas": oni_mes(8),
        "oni_amj": oni_mes(5),
    }).loc[1960:2026]
    pron_04 = {
        c: pd.read_csv(RES / f"pronosticos_rolling_{c}.csv").set_index("campania")
        for c in ("soja", "maiz", "trigo")
    }
    return enso, Y, X, pron_04


def test_rolling_h1_no_usa_rendimientos_t_ni_futuros():
    _, Y, X, pron_04 = _cargar_tablas()
    t_eval = 2010

    orig, _ = rolling_extensiones_h1(
        Y, X, pron_04, ["soja", "maiz", "trigo"], [t_eval], include_prophet=False
    )

    # Corromper rendimiento en t_eval y años posteriores, y ONI estricto posterior a t_eval
    Y_pert = Y.copy()
    Y_pert.loc[t_eval:, :] = Y_pert.loc[t_eval:, :] * 5.0
    X_pert = X.copy()
    X_pert.loc[t_eval + 1 :, :] = 9.9

    pert, _ = rolling_extensiones_h1(
        Y_pert, X_pert, pron_04, ["soja", "maiz", "trigo"], [t_eval], include_prophet=False
    )

    cols_pred = [
        "M0b RW + drift",
        "Híbrido RF (solo ONI)",
        "Híbrido GBM (solo ONI)",
        "M3_asim ARIMAX",
    ]
    for c in ("soja", "maiz", "trigo"):
        for col in cols_pred:
            assert orig[c].loc[t_eval, col] == pert[c].loc[t_eval, col], (
                f"Fuga detectada en {c} {col} para t={t_eval}"
            )


def test_rolling_h2_solo_entrena_antes_de_t_menos_1():
    _, Y, X, _ = _cargar_tablas()
    t_eval = 2012

    orig = rolling_arimax_h2(Y["maiz"], X["oni_jas"], [t_eval], cultivo="maiz")

    # Corromper explícitamente t_eval - 1 y t_eval en adelante (para h=2 no debe entrar t_eval - 1 al train)
    Y_pert = Y.copy()
    Y_pert.loc[t_eval - 1 :, "maiz"] = Y_pert.loc[t_eval - 1 :, "maiz"] * 10.0
    X_pert = X.copy()
    X_pert.loc[t_eval + 1 :, "oni_jas"] = -8.0

    pert = rolling_arimax_h2(Y_pert["maiz"], X_pert["oni_jas"], [t_eval], cultivo="maiz")

    assert orig.loc[t_eval, "M2_h2"] == pert.loc[t_eval, "M2_h2"]
    assert orig.loc[t_eval, "M3_h2"] == pert.loc[t_eval, "M3_h2"]


def test_puente_n34_usa_hasta_septiembre_de_s_como_maximo():
    enso, Y, _, pron_04 = _cargar_tablas()
    s_eval = 2005
    n34_orig = enso["n34_anom"].dropna()
    n34_pert = n34_orig.copy()
    corte = pd.Timestamp(f"{s_eval}-09-01")
    n34_pert.loc[n34_pert.index > corte] = 50.0

    hat_orig = pronosticar_puente_n34_djf(n34_orig, [s_eval])
    hat_pert = pronosticar_puente_n34_djf(n34_pert, [s_eval])
    assert hat_orig.loc[s_eval] == hat_pert.loc[s_eval]

    # También comprobar sin fuga en el paso 3b de rindes para t_eval
    serie_base = pronosticar_puente_n34_djf(n34_orig, range(1970, s_eval + 1))
    serie_pert = serie_base.copy()
    serie_pert.loc[serie_pert.index > s_eval] = -25.0
    Y_pert = Y.copy()
    Y_pert.loc[s_eval:, "soja"] = Y_pert.loc[s_eval:, "soja"] * 4.0

    p_base = rolling_puente_rindes(Y["soja"], serie_base, pron_04["soja"], [s_eval], cultivo="soja")
    p_pert = rolling_puente_rindes(Y_pert["soja"], serie_pert, pron_04["soja"], [s_eval], cultivo="soja")
    assert p_base.loc[s_eval] == p_pert.loc[s_eval]
