"""Tests unitarios y de equivalencia analítica para utils_ts.dm_test."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from utils_ts import dm_test


def test_dm_h1_coincide_con_formula_clasica():
    rng = np.random.default_rng(42)
    e1 = rng.normal(0.0, 0.8, size=30)
    e2 = rng.normal(0.0, 1.2, size=30)

    r_rect = dm_test(e1, e2, h=1, alternativa="two-sided", kernel="rectangular")
    r_bart = dm_test(e1, e2, h=1, alternativa="two-sided", kernel="bartlett")
    assert r_rect["DM"] == pytest.approx(r_bart["DM"], abs=1e-12)
    assert r_rect["p"] == pytest.approx(r_bart["p"], abs=1e-12)

    d = e1 ** 2 - e2 ** 2
    n = len(d)
    dbar = d.mean()
    gam0 = np.sum((d - dbar) ** 2) / n
    stat_esperado = (dbar / np.sqrt(gam0 / n)) * np.sqrt((n + 1 - 2 + 0) / n)
    p_esperado = 2.0 * stats.t.sf(abs(stat_esperado), df=n - 1)
    assert r_rect["DM"] == pytest.approx(stat_esperado, abs=1e-12)
    assert r_rect["p"] == pytest.approx(p_esperado, abs=1e-12)

    r_less = dm_test(e1, e2, h=1, alternativa="less", kernel="rectangular")
    assert r_less["p"] == pytest.approx(stats.t.cdf(stat_esperado, df=n - 1), abs=1e-12)


@pytest.mark.parametrize("h", [1, 4])
@pytest.mark.parametrize("kernel", ["rectangular", "bartlett"])
def test_dm_antisimetria_intercambiando_e1_e2(h, kernel):
    rng = np.random.default_rng(7)
    e1 = rng.normal(0.0, 0.9, size=40)
    e2 = rng.normal(0.0, 1.3, size=40)

    r12 = dm_test(e1, e2, h=h, alternativa="two-sided", kernel=kernel)
    r21 = dm_test(e2, e1, h=h, alternativa="two-sided", kernel=kernel)
    assert r12["DM"] == pytest.approx(-r21["DM"], abs=1e-12)
    assert r12["p"] == pytest.approx(r21["p"], abs=1e-12)

    p_less_12 = dm_test(e1, e2, h=h, alternativa="less", kernel=kernel)["p"]
    p_less_21 = dm_test(e2, e1, h=h, alternativa="less", kernel=kernel)["p"]
    assert p_less_12 + p_less_21 == pytest.approx(1.0, abs=1e-12)


def test_dm_varianza_bartlett_positiva_cuando_rectangular_falla():
    # Serie alternante d_t = e1^2 - e2^2 con autocovarianzas negativas en k=1,3
    n = 32
    d_target = np.array([1.2 if i % 2 == 0 else 0.0 for i in range(n)])
    e1 = np.sqrt(d_target + 1.0)
    e2 = np.ones(n)

    r_rect = dm_test(e1, e2, h=4, alternativa="less", kernel="rectangular")
    r_bart = dm_test(e1, e2, h=4, alternativa="less", kernel="bartlett")

    assert np.isnan(r_rect["DM"])
    assert np.isfinite(r_bart["DM"])
    assert np.isfinite(r_bart["p"])
    assert 0.0 <= r_bart["p"] <= 1.0


def test_dm_vector_analitico_bartlett_equivalente_a_r_forecast():
    """Verifica el cálculo cerrado de `dm_test` frente a fórmula analítica para h=3.

    Nota sobre R (`forecast::dm.test`): `Rscript` / el paquete `forecast` no está
    instalado en este entorno, por lo que no se ejecuta R en vivo. Matemáticamente,
    `forecast::dm.test(e1, e2, h=h, varestimator="bartlett")` computa:
      1. `gam_k = (1/n) * sum((d[k:] - dbar) * (d[:n-k] - dbar))` (`acf(..., type="covariance")`),
      2. `var = (gam[0] + 2 * sum((1 - k/h) * gam[k] for k in 1..h-1)) / n`,
      3. idéntico factor de Harvey-Leybourne-Newbold (1997):
         `hln = sqrt((n + 1 - 2*h + h*(h - 1)/n) / n)`,
      4. y evalúa `stat = (dbar / sqrt(var)) * hln` contra `t` con `df = n - 1`
         (el factor HLN es exactamente el mismo; difiere solo si `varestimator="acf"`).
    """
    e1 = np.array([1.0, 2.0, 1.0, 0.5, 1.5, 1.0, 2.0, 1.0])
    e2 = np.array([2.0, 1.5, 2.0, 1.5, 2.0, 1.5, 1.5, 2.0])
    h = 3

    d = e1 ** 2 - e2 ** 2
    n = len(d)
    dbar = d.mean()
    gam = [np.sum((d[k:] - dbar) * (d[: n - k] - dbar)) / n for k in range(h)]

    var_bart = (gam[0] + 2.0 * sum((1.0 - k / h) * gam[k] for k in range(1, h))) / n
    var_rect = (gam[0] + 2.0 * sum(gam[1:])) / n
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)

    stat_bart = (dbar / np.sqrt(var_bart)) * hln
    p_bart_2s = 2.0 * stats.t.sf(abs(stat_bart), df=n - 1)

    res_bart = dm_test(e1, e2, h=h, alternativa="two-sided", kernel="bartlett")
    assert res_bart["n"] == n
    assert res_bart["DM"] == pytest.approx(stat_bart, abs=1e-12)
    assert res_bart["p"] == pytest.approx(p_bart_2s, abs=1e-12)

    stat_rect = (dbar / np.sqrt(var_rect)) * hln
    res_rect = dm_test(e1, e2, h=h, alternativa="two-sided", kernel="rectangular")
    assert res_rect["DM"] == pytest.approx(stat_rect, abs=1e-12)


def test_dm_kernel_invalido():
    with pytest.raises(ValueError, match="kernel inválido"):
        dm_test([1.0, 2.0], [1.5, 2.5], kernel="desconocido")
