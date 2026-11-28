"""Tests de integridad y convenciones temporales para data/processed y src/01_preprocesamiento.py."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
PROC = RAIZ / "data" / "processed"
sys.path.insert(0, str(RAIZ / "src"))

from utils_ts import oni_campanias, rindes


def _cargar_preprocesamiento():
    spec = importlib.util.spec_from_file_location("preproc", RAIZ / "src" / "01_preprocesamiento.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parquets_sin_valores_infinitos():
    parquets = sorted(PROC.glob("*.parquet"))
    assert len(parquets) >= 4
    for path in parquets:
        df = pd.read_parquet(path)
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            arr = df[col].to_numpy(dtype=float)
            assert not np.isinf(arr).any(), f"Se encontró inf en {path.name} columna {col}"


def test_serie_pampeana_1970_2024_completa_tres_cultivos():
    rend = pd.read_parquet(PROC / "rendimientos_anuales.parquet")
    Y = rindes(rend, region="pampeana", inicio=1970, fin=2024)
    assert list(Y.columns) == ["soja", "maiz", "trigo"]
    assert Y.index.min() == 1970
    assert Y.index.max() == 2024
    assert len(Y) == 55
    assert not Y.isna().any().any()
    assert (Y > 0).all().all()


def test_mapeo_oni_campanias_jas_0_y_amj_1():
    enso = pd.read_parquet(PROC / "enso_mensual.parquet")
    camp = pd.read_parquet(PROC / "campanias.parquet")
    oni_df = oni_campanias(camp)

    for s in range(1970, 2025):
        jas_camp = float(oni_df.loc[s, "oni_JAS_0"])
        ago_enso = float(enso.loc[pd.Timestamp(s, 8, 1), "oni"])
        assert np.isclose(jas_camp, ago_enso, atol=1e-12), f"Año {s}: JAS_0={jas_camp} != ago={ago_enso}"

        amj1_camp = float(oni_df.loc[s, "oni_AMJ_1"])
        may_next_enso = float(enso.loc[pd.Timestamp(s + 1, 5, 1), "oni"])
        assert np.isclose(amj1_camp, may_next_enso, atol=1e-12), f"Año {s}: AMJ_1={amj1_camp} != may+1={may_next_enso}"


def test_leer_rendimientos_min_count_evita_inf_1923():
    preproc = _cargar_preprocesamiento()
    rend, _ = preproc.leer_rendimientos()
    num_cols = rend.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        assert not np.isinf(rend[col].to_numpy(dtype=float)).any()
    fila_1923 = rend[(rend.cultivo == "maiz") & (rend.region == "pampeana") & (rend.campania == 1923)]
    assert len(fila_1923) == 1
    assert pd.isna(fila_1923["rend_kgha"].iloc[0])
