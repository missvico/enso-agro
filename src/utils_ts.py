"""Funciones compartidas: estilo de gráficos, tests de estacionariedad, ccf, tendencias."""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.stattools import adfuller, kpss

# ------------------------------------------------------------------ estilo
COLOR = {
    "nino": "#e34948",     # fase cálida
    "nina": "#2a78d6",     # fase fría
    "neutral": "#b5b3ad",
    "tinta": "#0b0b0b",
    "tinta2": "#52514e",
    "grilla": "#e4e3df",
    "serie": "#2a78d6",
    "tendencia": "#eb6834",
}
COLOR["fondo"] = "#fcfcfb"

# Tema alternativo para el deck (paleta de la presentación del equipo): TEMA_FIG=salvia
TEMA = os.environ.get("TEMA_FIG", "claro")
if TEMA == "salvia":
    COLOR.update({"fondo": "#E8F0E2", "tinta": "#0B1A15", "tinta2": "#3F5248", "grilla": "#D2DFD0",
                  "neutral": "#A3B2A6", "tendencia": "#C4622D"})
CULTIVO_COLOR = {"soja": "#2a78d6", "maiz": "#eb6834", "trigo": "#1baf7a"}
CULTIVO_NOMBRE = {"soja": "Soja", "maiz": "Maíz", "trigo": "Trigo"}


def estilo() -> None:
    familia = {}
    if TEMA == "salvia":
        fuentes = Path(__file__).resolve().parents[1] / "assets" / "fonts"
        for f in fuentes.glob("*.ttf"):
            mpl.font_manager.fontManager.addfont(str(f))
        # DejaVu Sans como respaldo glifo a glifo (ρ, ⁺, flechas) que Familjen Grotesk no trae
        familia = {"font.family": ["Familjen Grotesk", "DejaVu Sans"], "axes.titleweight": "medium", "axes.titlesize": 13}
    mpl.rcParams.update({
        "figure.figsize": (10, 4.2),
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "figure.constrained_layout.use": True,
        "axes.facecolor": COLOR["fondo"],
        "figure.facecolor": COLOR["fondo"],
        "axes.edgecolor": COLOR["grilla"],
        "axes.labelcolor": COLOR["tinta2"],
        "axes.titlecolor": COLOR["tinta"],
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": COLOR["grilla"],
        "grid.linewidth": 0.6,
        "xtick.color": COLOR["tinta2"],
        "ytick.color": COLOR["tinta2"],
        "lines.linewidth": 1.6,
        "legend.frameon": False,
        "font.size": 10,
        **familia,
    })


# ------------------------------------------------------------------ estacionariedad
def adf_kpss(x: pd.Series, regresion: str = "c") -> dict:
    """ADF (H0: raíz unitaria) + KPSS (H0: estacionaria) con la tabla de decisión de la Clase 11."""
    x = pd.Series(x).dropna()
    p_adf = adfuller(x, regression=regresion, autolag="AIC")[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # KPSS avisa cuando el p-valor cae fuera de su tabla
        p_kpss = kpss(x, regression=regresion, nlags="auto")[1]
    if p_adf < 0.05 and p_kpss >= 0.05:
        decision = "estacionaria"
    elif p_adf >= 0.05 and p_kpss < 0.05:
        decision = "no estacionaria"
    else:
        decision = "indefinida"
    return {"n": len(x), "p_ADF": round(p_adf, 3), "p_KPSS": round(p_kpss, 3), "decisión": decision}


# ------------------------------------------------------------------ correlación cruzada
def ccf(x: pd.Series, y: pd.Series, max_lag: int = 12) -> pd.DataFrame:
    """r(k) = corr(x_t, y_{t+k}). k > 0  =>  x adelanta a y."""
    filas = []
    for k in range(-max_lag, max_lag + 1):
        par = pd.concat([x, y.shift(-k)], axis=1).dropna()
        filas.append({"lag": k, "r": par.iloc[:, 0].corr(par.iloc[:, 1]), "n": len(par)})
    df = pd.DataFrame(filas)
    df["umbral95"] = 1.96 / np.sqrt(df["n"])
    return df


# ------------------------------------------------------------------ tendencias
def tendencia(y: pd.Series, metodo: str = "lowess", frac: float = 0.35, ventana: int = 9) -> pd.Series:
    """Tendencia tecnológica de una serie anual de rendimientos."""
    y = y.dropna()
    t = y.index.to_numpy(dtype=float)
    if metodo == "lineal":
        b = np.polyfit(t, y.to_numpy(), 1)
        return pd.Series(np.polyval(b, t), index=y.index)
    if metodo == "cuadratica":
        b = np.polyfit(t, y.to_numpy(), 2)
        return pd.Series(np.polyval(b, t), index=y.index)
    if metodo == "lowess":
        return pd.Series(lowess(y.to_numpy(), t, frac=frac, it=2, return_sorted=False), index=y.index)
    if metodo == "ma":
        # MA centrada (Clase 4); en los bordes la ventana se achica (min_periods)
        return y.rolling(ventana, center=True, min_periods=ventana // 2 + 1).mean()
    raise ValueError(metodo)


def anomalia_relativa(y: pd.Series, tend: pd.Series) -> pd.Series:
    """Desvío porcentual respecto de la tendencia: 100 * (y / tendencia - 1)."""
    return 100 * (y / tend - 1)


# ------------------------------------------------------------------ tablas por campaña
INICIO, FIN = 1970, 2024
VENTANA_FENO = {  # ventanas fijadas a priori (Etapa 1)
    "soja": ["oni_OND_0", "oni_NDJ_0", "oni_DJF_1"],
    "maiz": ["oni_OND_0", "oni_NDJ_0", "oni_DJF_1"],
    "trigo": ["oni_JAS_0", "oni_ASO_0", "oni_SON_0"],
}


def rindes(rend: pd.DataFrame, region: str = "pampeana", columna: str = "rend_kgha",
           inicio: int = INICIO, fin: int = FIN) -> pd.DataFrame:
    """Rendimientos en formato ancho (campaña x cultivo)."""
    r = rend[(rend.region == region) & rend.campania.between(inicio, fin)]
    return r.pivot(index="campania", columns="cultivo", values=columna)[["soja", "maiz", "trigo"]]


def anomalias(Y: pd.DataFrame, metodo: str = "lowess") -> pd.DataFrame:
    """Anomalía relativa (%) de cada cultivo respecto de su tendencia."""
    return pd.DataFrame({c: anomalia_relativa(Y[c], tendencia(Y[c], metodo)) for c in Y})


def oni_campanias(camp: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in camp.columns if c.startswith("oni_")]
    return camp.drop_duplicates("campania").set_index("campania")[cols]


# ------------------------------------------------------------------ evaluación de pronósticos
def dm_test(e1, e2, h: int = 1, alternativa: str = "two-sided") -> dict:
    """Diebold-Mariano (1995) con la corrección de Harvey, Leybourne y Newbold (1997).

    e1, e2: errores de pronóstico de los modelos 1 y 2 sobre los mismos objetivos.
    Pérdida cuadrática. d = e1² − e2²; DM < 0 => el modelo 1 es más preciso.
    alternativa: 'two-sided' o 'less' (H1: el modelo 1 tiene menor pérdida).
    """
    from scipy import stats as _st
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    ok = ~(np.isnan(e1) | np.isnan(e2))
    d = e1[ok] ** 2 - e2[ok] ** 2
    n = len(d)
    dbar = d.mean()
    gam = [np.sum((d[k:] - dbar) * (d[:n - k] - dbar)) / n for k in range(h)]
    var = (gam[0] + 2 * sum(gam[1:])) / n
    dm = dbar / np.sqrt(var) if var > 0 else np.nan
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    if alternativa == "less":
        p = _st.t.cdf(stat, df=n - 1)
    else:
        p = 2 * _st.t.sf(abs(stat), df=n - 1)
    return {"n": n, "DM": stat, "p": p}


def metricas(y, yhat) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = ~(np.isnan(y) | np.isnan(yhat))
    e = y[ok] - yhat[ok]
    return {"n": int(ok.sum()), "RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e))),
            "r": float(np.corrcoef(y[ok], yhat[ok])[0, 1]) if ok.sum() > 2 else np.nan}
