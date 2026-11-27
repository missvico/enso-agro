# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Etapa 4 — Extensiones analíticas: desambiguación de baselines, asimetría ENSO, puente 3a→3b, DM+Holm y coherencia wavelet
#
# **TP Data Mining de Series Temporales — MEDyDC, FCEyN-UBA**
#
# Esta notebook cierra las brechas metodológicas identificadas en la auditoría (`AUDIT_V01.md`) sin modificar
# los artefactos congelados de las etapas `01` a `04`:
#
# | Bloque | Qué resuelve | Salidas |
# |---|---|---|
# | 1. Desambiguación de baselines | `RW+drift`, `Prophet sin ONI` y `Híbridos RF/GBM (solo ONI)` en rolling `1995–2024` | Tabla comparativa |
# | 2. Asimetría Niña vs. Niño | Test de Wald + bootstrap en H1, `ARIMAX asimétrico` rolling y sensibilidad `2026/27` | `17_asimetria_nina_nino.png` |
# | 3. Rigor causal, `h=2` y puente 3a→3b | Trigo causal en `AMJ`, rolling `h=2` (`1996–2024`) y `M3_puente` con `Niño 3.4` pronosticado desde sep | `18_baselines_y_puente_3a_3b.png` |
# | 4. Diagnósticos + DM unificado y Holm | ACF/PACF de `Δlog`, Ljung-Box y tabla maestra `p_DM (log)` vs `p_DM (kg/ha)` + Holm | `19_diagnosticos_y_dm.png` |
# | 5. Coherencia wavelet (Grinsted 2004) | `R²(s, t)` Morlet (`ω0=6`) entre `Niño 3.4` y `precip_z` (`1981–2025`), COI, fase y Monte Carlo AR(1) | `20_coherencia_wavelet.png` |

# %%
import ast
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from prophet import Prophet
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.arima.model import ARIMA

RAIZ = Path.cwd().resolve()
if RAIZ.name == "notebooks":
    RAIZ = RAIZ.parent
sys.path.insert(0, str(RAIZ / "src"))
from utils_ts import (COLOR, CULTIVO_COLOR, CULTIVO_NOMBRE, anomalias, dm_test, estilo,
                      oni_campanias, rindes)

warnings.filterwarnings("ignore")
for lg in ("cmdstanpy", "prophet"):
    logging.getLogger(lg).setLevel(logging.CRITICAL)
    logging.getLogger(lg).disabled = True

np.random.seed(0)
estilo()
PROC = RAIZ / "data" / "processed"
FIG = RAIZ / os.environ.get("FIG_DIR", "figures")
RES = RAIZ / "results"
FIG.mkdir(exist_ok=True, parents=True)
RES.mkdir(exist_ok=True, parents=True)

enso = pd.read_parquet(PROC / "enso_mensual.parquet")
clima = pd.read_parquet(PROC / "clima_pampeano_mensual.parquet")
rend = pd.read_parquet(PROC / "rendimientos_anuales.parquet")
camp = pd.read_parquet(PROC / "campanias.parquet")

pron_04 = {c: pd.read_csv(RES / f"pronosticos_rolling_{c}.csv").set_index("campania")
           for c in ("soja", "maiz", "trigo")}
etapa3b = json.loads((RES / "etapa3b_rindes.json").read_text(encoding="utf-8"))

Y = rindes(rend)
O = oni_campanias(camp)
jas = lambda col: enso[col].groupby(enso.index.year).apply(lambda s: s[s.index.month.isin([7, 8, 9])].mean())
oni_mes = lambda mes, desfase=0: enso["oni"][enso.index.month == mes].set_axis(
    enso.index[enso.index.month == mes].year - desfase
)
X = pd.DataFrame({
    "oni_jas": oni_mes(8),
    "oni_amj": oni_mes(5),
    "oni_tend": oni_mes(8) - oni_mes(5),
    "soi_jas": jas("soi"),
    "iod_jas": jas("iod"),
    "oni_djf": oni_mes(1, desfase=1),
}).loc[1960:2026]
assert np.allclose(X.loc[1970:2024, "oni_jas"], O.loc[1970:2024, "oni_JAS_0"])
assert np.allclose(X.loc[1970:2024, "oni_amj"], O.loc[1970:2024, "oni_AMJ_0"])

CULTIVOS = ["soja", "maiz", "trigo"]
TEST = list(range(1995, 2025))
nina_mask = (X.loc[TEST, "oni_jas"] <= -0.5).to_numpy()
ext_results = {}


def aicc(res) -> float:
    k = len(res.params)
    return res.aic + 2 * k * (k + 1) / max(res.nobs - k - 1, 1)


def arima_auto(ly: pd.Series, exog=None):
    """auto.arima reducido idéntico a Etapa 3b: (p,1,q) con drift, p,q ∈ {0,1,2}."""
    mejor = None
    for p in range(3):
        for q in range(3):
            try:
                r = ARIMA(ly, exog=exog, order=(p, 1, q), trend="t").fit()
            except Exception:
                continue
            if mejor is None or aicc(r) < aicc(mejor[1]):
                mejor = ((p, 1, q), r)
    return mejor


def resumen_metricas(obs: pd.Series, pred: pd.Series, m2_ref: pd.Series, mask_nina: np.ndarray) -> dict:
    e = obs.to_numpy(float) - pred.to_numpy(float)
    e_ref = obs.to_numpy(float) - m2_ref.to_numpy(float)
    mse_ref = np.mean(e_ref ** 2)
    return {
        "RMSE kg/ha": float(np.sqrt(np.mean(e ** 2))),
        "MAE kg/ha": float(np.mean(np.abs(e))),
        "MAPE %": float(100 * np.mean(np.abs(e / obs.to_numpy(float)))),
        "skill vs ARIMA %": float(100 * (1 - np.mean(e ** 2) / mse_ref)),
        "RMSE años Niña": float(np.sqrt(np.mean(e[mask_nina] ** 2))),
    }


print(f"Datos listos · TEST = {TEST[0]}..{TEST[-1]} (n={len(TEST)}, años Niña={int(nina_mask.sum())})")

# %% [markdown]
# ## Bloque 1 — Desambiguación de Baselines (Etapa 3b, rolling `1995–2024`, `n = 30`)
#
# Separamos dos efectos que quedaron mezclados en la Etapa 3b original:
# 1. **Prophet sin ONI (`M5_sin_oni`) vs `M5 Prophet + ONI`:** cuantifica cuánto del skill de Prophet en maíz
#    (`+35,2 %`) y trigo (`+14,9 %`) proviene de su tendencia lineal por tramos (quiebre ~2015) y cuánto del ONI.
# 2. **Híbridos `RF` y `GBM` solo con `oni_jas` (5 semillas `0..4`) vs `Híbridos` de 4 exógenas y `M3 ARIMAX`:**
#    aísla el aporte de la no-linealidad respecto de sumar `oni_tend`, `soi_jas` e `iod_jas`.
# 3. **`M0b: Random Walk + drift`** como cota mínima con tendencia estocástica.

# %%
t0 = time.time()
ext_rolling = {}

for c in CULTIVOS:
    ly_all = np.log(Y[c])
    filas = []
    for t in TEST:
        tr = ly_all.loc[:t - 1]
        anios = tr.index
        ly = tr.reset_index(drop=True)
        orden = ast.literal_eval(pron_04[c].loc[t, "orden"])
        r2 = ARIMA(ly, order=orden, trend="t").fit()
        pred_m2 = float(np.exp(r2.forecast(1).iloc[0]))
        assert np.isclose(pred_m2, pron_04[c].loc[t, "M2 ARIMA"], rtol=2e-2)

        # M0b: Random Walk + drift en log
        drift = float(ly.diff().dropna().mean())
        pred_rw_drift = float(np.exp(ly.iloc[-1] + drift))

        # M5_sin_oni: Prophet univariado (misma config que M5 sin regresor)
        dfp = pd.DataFrame({"ds": pd.to_datetime(anios.astype(str)), "y": ly.to_numpy()})
        mp = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
                     changepoint_prior_scale=0.05, uncertainty_samples=0)
        mp.fit(dfp)
        pred_prophet_sin = float(np.exp(mp.predict(pd.DataFrame({"ds": [pd.Timestamp(f"{t}")]}))["yhat"].iloc[0]))

        # Híbridos RF y GBM solo con oni_jas (promedio 5 semillas 0..4)
        resid = r2.resid.iloc[1:].to_numpy()
        X_oni_tr = X.loc[anios[1:], ["oni_jas"]].to_numpy()
        X_oni_te = X.loc[[t], ["oni_jas"]].to_numpy()
        rf_preds = [RandomForestRegressor(n_estimators=400, max_depth=3, min_samples_leaf=4, random_state=s)
                    .fit(X_oni_tr, resid).predict(X_oni_te)[0] for s in range(5)]
        gbm_preds = [GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.03,
                                               subsample=0.8, min_samples_leaf=4, random_state=s)
                     .fit(X_oni_tr, resid).predict(X_oni_te)[0] for s in range(5)]

        # M3_asim: ARIMAX asimétrico (Bloque 2 rolling)
        oni_tr_arr = X.loc[anios, "oni_jas"].to_numpy()
        ex_asim_tr = np.column_stack([np.maximum(oni_tr_arr, 0.0), np.minimum(oni_tr_arr, 0.0)])
        oni_t_val = float(X.loc[t, "oni_jas"])
        ex_asim_te = np.array([[max(oni_t_val, 0.0), min(oni_t_val, 0.0)]])
        r_asim = ARIMA(ly, exog=ex_asim_tr, order=orden, trend="t").fit()
        pred_asim = float(np.exp(r_asim.forecast(1, exog=ex_asim_te).iloc[0]))

        filas.append({
            "cultivo": c,
            "campania": t,
            "obs": float(Y.loc[t, c]),
            "M2 ARIMA": pred_m2,
            "M0b RW + drift": pred_rw_drift,
            "M5_sin_oni Prophet": pred_prophet_sin,
            "Híbrido RF (solo ONI)": float(pred_m2 * np.exp(np.mean(rf_preds))),
            "Híbrido GBM (solo ONI)": float(pred_m2 * np.exp(np.mean(gbm_preds))),
            "M3_asim ARIMAX": pred_asim,
        })
    ext_rolling[c] = pd.DataFrame(filas).set_index("campania")

print(f"Bloque 1+2 rolling listo en {time.time() - t0:.1f} s", flush=True)

# %%
modelos_b1 = ["M0b RW + drift", "M2 ARIMA", "M5_sin_oni Prophet", "M5 Prophet + ONI",
              "M3 ARIMAX (ONI jul–sep)", "Híbrido RF (solo ONI)", "Híbrido RF",
              "Híbrido GBM (solo ONI)", "Híbrido GBM"]
tabla_b1 = []
for c in CULTIVOS:
    p04 = pron_04[c]
    pe = ext_rolling[c]
    obs, m2 = p04["obs"], p04["M2 ARIMA"]
    for mdl in modelos_b1:
        serie = pe[mdl] if mdl in pe.columns else p04[mdl]
        mt = resumen_metricas(obs, serie, m2, nina_mask)
        tabla_b1.append({"cultivo": c, "modelo": mdl, **mt})
tabla_b1_df = pd.DataFrame(tabla_b1)
for c in CULTIVOS:
    print(f"\n=== Bloque 1 · {CULTIVO_NOMBRE[c]} (n = 30) ===")
    print(tabla_b1_df[tabla_b1_df.cultivo == c].drop(columns="cultivo").set_index("modelo").round(2).to_string())

# Descomposición explícita de Prophet (M5 + ONI vs M5_sin_oni)
descomp_prophet = []
for c in CULTIVOS:
    p04 = pron_04[c]
    pe = ext_rolling[c]
    obs = p04["obs"]
    rmse_sin = float(np.sqrt(np.mean((obs - pe["M5_sin_oni Prophet"]) ** 2)))
    rmse_con = float(np.sqrt(np.mean((obs - p04["M5 Prophet + ONI"]) ** 2)))
    dm_log = dm_test(np.log(obs) - np.log(p04["M5 Prophet + ONI"]),
                     np.log(obs) - np.log(pe["M5_sin_oni Prophet"]), h=1, alternativa="less")
    dm_kg = dm_test(obs - p04["M5 Prophet + ONI"], obs - pe["M5_sin_oni Prophet"], h=1, alternativa="less")
    descomp_prophet.append({
        "cultivo": c, "RMSE_sin_ONI": round(rmse_sin, 2), "RMSE_con_ONI": round(rmse_con, 2),
        "delta_RMSE": round(rmse_con - rmse_sin, 2),
        "p_DM_con_vs_sin_log": round(float(dm_log["p"]), 4),
        "p_DM_con_vs_sin_kg": round(float(dm_kg["p"]), 4),
    })
print("\nProphet + ONI vs Prophet sin ONI (¿aporta ONI sobre la tendencia por tramos?):")
print(pd.DataFrame(descomp_prophet).to_string(index=False))

ext_results["bloque1_baselines"] = tabla_b1_df.round(3).to_dict("records")
ext_results["bloque1_prophet_descomposicion"] = descomp_prophet

# %% [markdown]
# ## Bloque 2 — Test Formal de Asimetría Niña vs. Niño
#
# Evaluamos si la respuesta de los rendimientos al ONI pre-siembra es simétrica (`H0: β_pos = β_neg`) en:
# 1. **Inferencia H1 (`1971–2024`, `n = 54`):** `anom_t = α + φ·anom_{t-1} + β_pos·max(ONI_JAS, 0) + β_neg·min(ONI_JAS, 0) + ε_t`
#    con errores robustos `HC3` y bootstrap de 2.000 réplicas.
# 2. **Pronóstico rolling (`1995–2024`):** `M3_asim: ARIMAX asimétrico`.
# 3. **Sensibilidad `2026/27` (`ONI JAS = +2,16 °C`):** contraste lineal vs quebrado y límite de extrapolación.

# %%
A = anomalias(Y, "lowess")
asim_h1 = {}
rng_boot = np.random.default_rng(0)

for c in CULTIVOS:
    d = pd.DataFrame({
        "y": A[c],
        "y1": A[c].shift(1),
        "oni": X.loc[1970:2024, "oni_jas"],
    }).dropna()
    d["oni_pos"] = np.maximum(d["oni"], 0.0)
    d["oni_neg"] = np.minimum(d["oni"], 0.0)

    r_hc3 = smf.ols("y ~ y1 + oni_pos + oni_neg", d).fit(cov_type="HC3")
    w = r_hc3.wald_test("oni_pos - oni_neg = 0", scalar=True)

    b_pos_boot, b_neg_boot, diff_boot = [], [], []
    arr = d[["y", "y1", "oni_pos", "oni_neg"]].to_numpy()
    n_obs = len(arr)
    for _ in range(2000):
        idx_b = rng_boot.integers(0, n_obs, size=n_obs)
        sb = arr[idx_b]
        Xb = np.column_stack([np.ones(n_obs), sb[:, 1], sb[:, 2], sb[:, 3]])
        coef, *_ = np.linalg.lstsq(Xb, sb[:, 0], rcond=None)
        b_pos_boot.append(coef[2])
        b_neg_boot.append(coef[3])
        diff_boot.append(coef[3] - coef[2])

    ic_pos_hc3 = r_hc3.conf_int().loc["oni_pos"].to_numpy()
    ic_neg_hc3 = r_hc3.conf_int().loc["oni_neg"].to_numpy()
    ic_pos_bt = np.percentile(b_pos_boot, [2.5, 97.5])
    ic_neg_bt = np.percentile(b_neg_boot, [2.5, 97.5])
    p_boot_2c = float(2 * min(np.mean(np.array(diff_boot) <= 0), np.mean(np.array(diff_boot) >= 0)))

    asim_h1[c] = {
        "n": int(n_obs),
        "beta_pos_%/°C": float(r_hc3.params["oni_pos"]),
        "p_pos_HC3": float(r_hc3.pvalues["oni_pos"]),
        "IC95_pos_HC3": [float(ic_pos_hc3[0]), float(ic_pos_hc3[1])],
        "IC95_pos_boot": [float(ic_pos_bt[0]), float(ic_pos_bt[1])],
        "beta_neg_%/°C": float(r_hc3.params["oni_neg"]),
        "p_neg_HC3": float(r_hc3.pvalues["oni_neg"]),
        "IC95_neg_HC3": [float(ic_neg_hc3[0]), float(ic_neg_hc3[1])],
        "IC95_neg_boot": [float(ic_neg_bt[0]), float(ic_neg_bt[1])],
        "diff_neg_menos_pos": float(r_hc3.params["oni_neg"] - r_hc3.params["oni_pos"]),
        "wald_stat_HC3": float(np.squeeze(w.statistic)),
        "p_wald_HC3": float(w.pvalue),
        "p_boot_2colas": min(p_boot_2c, 1.0),
    }

print("Asimetría en H1 (1971–2024, anomalía LOWESS %):")
print(pd.DataFrame({c: {k: v for k, v in d.items() if not isinstance(v, list)}
                    for c, d in asim_h1.items()}).T.round(4).to_string())

# Evaluación rolling de M3_asim vs M3 lineal y RF/GBM
asim_rolling = []
for c in CULTIVOS:
    p04, pe = pron_04[c], ext_rolling[c]
    obs, m2 = p04["obs"], p04["M2 ARIMA"]
    for mdl, serie in [("M3 ARIMAX (lineal)", p04["M3 ARIMAX (ONI jul–sep)"]),
                       ("M3_asim ARIMAX (quebrado)", pe["M3_asim ARIMAX"]),
                       ("Híbrido RF", p04["Híbrido RF"]),
                       ("Híbrido GBM", p04["Híbrido GBM"])]:
        mt = resumen_metricas(obs, serie, m2, nina_mask)
        asim_rolling.append({"cultivo": c, "modelo": mdl, **mt})
print("\nAsimetría en rolling (1995–2024):")
print(pd.DataFrame(asim_rolling).round(2).to_string(index=False))

# Sensibilidad 2026/27 (ONI JAS 2025 = -0.26, ONI JAS 2026 = +2.16)
sens_2026 = []
for c in ("soja", "maiz"):
    ly = np.log(Y[c]).reset_index(drop=True)
    orden, r2 = arima_auto(ly)
    oni_hist = X.loc[Y.index, "oni_jas"].to_numpy()
    oni_fut = X.loc[[2025, 2026], "oni_jas"].to_numpy()

    r_lin = ARIMA(ly, exog=oni_hist[:, None], order=orden, trend="t").fit()
    ex_asim_h = np.column_stack([np.maximum(oni_hist, 0.0), np.minimum(oni_hist, 0.0)])
    ex_asim_f = np.column_stack([np.maximum(oni_fut, 0.0), np.minimum(oni_fut, 0.0)])
    r_as = ARIMA(ly, exog=ex_asim_h, order=orden, trend="t").fit()

    f2 = r2.get_forecast(2)
    fl = r_lin.get_forecast(2, exog=oni_fut[:, None])
    fa = r_as.get_forecast(2, exog=ex_asim_f)
    for k, anio in enumerate([2025, 2026]):
        i80_a = np.exp(fa.conf_int(alpha=0.2).iloc[k])
        i95_a = np.exp(fa.conf_int(alpha=0.05).iloc[k])
        sens_2026.append({
            "cultivo": c,
            "campaña": f"{anio}/{str(anio + 1)[-2:]}",
            "ONI_JAS": float(oni_fut[k]),
            "ARIMA_kg_ha": float(np.exp(f2.predicted_mean.iloc[k])),
            "ARIMAX_lineal_kg_ha": float(np.exp(fl.predicted_mean.iloc[k])),
            "ARIMAX_asim_kg_ha": float(np.exp(fa.predicted_mean.iloc[k])),
            "IC80_asim": f"{i80_a.iloc[0]:.0f}–{i80_a.iloc[1]:.0f}",
            "IC95_asim": f"{i95_a.iloc[0]:.0f}–{i95_a.iloc[1]:.0f}",
            "beta_pos_%": float(100 * r_as.params.iloc[1]),
            "beta_neg_%": float(100 * r_as.params.iloc[2]),
            "delta_asim_vs_lin_%": float(100 * (np.exp(fa.predicted_mean.iloc[k]) / np.exp(fl.predicted_mean.iloc[k]) - 1)),
        })

print("\nSensibilidad 2025/26 y 2026/27 (lineal vs asimétrico):")
print(pd.DataFrame(sens_2026).round(2).to_string(index=False))

ext_results["bloque2_asimetria_h1"] = asim_h1
ext_results["bloque2_asimetria_rolling"] = pd.DataFrame(asim_rolling).round(3).to_dict("records")
ext_results["bloque2_sensibilidad_2026"] = pd.DataFrame(sens_2026).round(2).to_dict("records")

# %%
# Figura 17: asimetría Niña vs Niño en soja y maíz
fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
xx_neg = np.linspace(-1.4, 0.0, 40)
xx_pos = np.linspace(0.0, 2.25, 60)
xx_all = np.linspace(-1.4, 2.25, 100)

for ax, c in zip(axs, ("soja", "maiz")):
    d = pd.DataFrame({"y": A[c], "y1": A[c].shift(1), "oni": X.loc[1970:2024, "oni_jas"]}).dropna()
    d["oni_pos"] = np.maximum(d["oni"], 0.0)
    d["oni_neg"] = np.minimum(d["oni"], 0.0)
    r_lin_h1 = smf.ols("y ~ oni", d).fit()
    r_as_h1 = smf.ols("y ~ oni_pos + oni_neg", d).fit(cov_type="HC3")

    cols = [COLOR["nina"] if v <= -0.5 else (COLOR["nino"] if v >= 0.5 else COLOR["neutral"]) for v in d["oni"]]
    ax.scatter(d["oni"], d["y"], c=cols, s=30, edgecolor=COLOR["fondo"], lw=0.8, zorder=3)
    ax.plot(xx_all, r_lin_h1.params["Intercept"] + r_lin_h1.params["oni"] * xx_all,
            color=COLOR["tinta2"], lw=1.3, ls="--", label="Ajuste lineal")

    # Banda IC95% del ajuste quebrado (sobre y ~ oni_pos + oni_neg)
    grid_x = np.r_[xx_neg, xx_pos]
    X_grid = np.column_stack([np.ones_like(grid_x), np.maximum(grid_x, 0.0), np.minimum(grid_x, 0.0)])
    cov = r_as_h1.cov_params().to_numpy()
    yhat = X_grid @ r_as_h1.params.to_numpy()
    se = np.sqrt(np.einsum("ij,jk,ik->i", X_grid, cov, X_grid))
    ax.fill_between(grid_x, yhat - 1.96 * se, yhat + 1.96 * se, color=CULTIVO_COLOR[c], alpha=0.14, lw=0, label="IC95% quebrado")
    ax.plot(grid_x, yhat, color=CULTIVO_COLOR[c], lw=2.2, label="Ajuste quebrado (Niña / Niño)")

    ax.axvline(0, color=COLOR["tinta"], lw=0.7)
    ax.axhline(0, color=COLOR["tinta"], lw=0.7)
    ax.axvline(1.79, color=COLOR["tinta2"], lw=0.8, ls=":")
    ax.axvline(2.16, color=COLOR["nino"], lw=1.2, ls=":")
    ax.text(1.76, 25.5, "máx hist.\n+1,79", ha="right", va="top", fontsize=7.5, color=COLOR["tinta2"])
    ax.text(2.14, -36, "2026\n+2,16", ha="right", fontsize=7.5, color=COLOR["nino"], fontweight="bold")
    info = asim_h1[c]
    ax.set_title(
        f"{CULTIVO_NOMBRE[c]} · "
        + rf"$\beta_{{\mathrm{{Niña}}}}={info['beta_neg_%/°C']:+.1f}$ vs "
        + rf"$\beta_{{\mathrm{{Niño}}}}={info['beta_pos_%/°C']:+.1f}$ %/°C "
        + f"(Wald p={info['p_wald_HC3']:.3f})",
        fontsize=10,
    )
    ax.set_xlabel("ONI jul–sep (°C)")
axs[0].set_ylabel("anomalía de rendimiento (%)")
axs[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Asimetría Niña vs. Niño en la respuesta del rendimiento al ONI pre-siembra (1971–2024)",
             x=0.01, ha="left", fontweight="semibold")
fig.savefig(FIG / "17_asimetria_nina_nino.png")
plt.show()

# %% [markdown]
# ## Bloque 3 — Rigor Causal, Horizonte `h = 2` y Puente Etapa 3a → 3b
#
# 1. **Trigo causal (`oni_AMJ_0`):** reemplaza `oni_jas` (posterior a la siembra jun–jul del trigo) por `oni_amj` (abr–jun).
# 2. **Rolling `h = 2` (`1996–2024`, `n = 29`):** para cada `t`, entrena hasta `t-2` y pronostica `t` a 2 pasos,
#    validando el horizonte real utilizado para proyectar `2026/27` desde el último dato `2024/25`.
# 3. **Puente explícito Etapa 3a → 3b (`M3_puente`):** para cada año `s ∈ [1970, 2024]`, ajusta `ARIMA(2,0,2)`
#    sobre `n34_anom` mensual hasta `s-09-01` y pronostica `DJF` (`h = 3, 4, 5`). Ese `n34_djf_hat` alimenta al ARIMAX.

# %%
# 3.1 Trigo causal en oni_AMJ_0
trigo_amj_filas = []
ly_trigo = np.log(Y["trigo"])
for t in TEST:
    tr = ly_trigo.loc[:t - 1]
    anios = tr.index
    ly = tr.reset_index(drop=True)
    orden = ast.literal_eval(pron_04["trigo"].loc[t, "orden"])
    r_amj = ARIMA(ly, exog=X.loc[anios, ["oni_amj"]].to_numpy(), order=orden, trend="t").fit()
    pred_amj = float(np.exp(r_amj.forecast(1, exog=X.loc[[t], ["oni_amj"]].to_numpy()).iloc[0]))
    trigo_amj_filas.append({"campania": t, "M3_trigo_AMJ": pred_amj})
df_trigo_amj = pd.DataFrame(trigo_amj_filas).set_index("campania")
mt_trigo_amj = resumen_metricas(pron_04["trigo"]["obs"], df_trigo_amj["M3_trigo_AMJ"], pron_04["trigo"]["M2 ARIMA"], nina_mask)
dm_trigo_amj = dm_test(np.log(pron_04["trigo"]["obs"]) - np.log(df_trigo_amj["M3_trigo_AMJ"]),
                       np.log(pron_04["trigo"]["obs"]) - np.log(pron_04["trigo"]["M2 ARIMA"]), h=1, alternativa="less")
print("3.1 Trigo causal con ONI AMJ (previo a siembra jun–jul):")
print({**{k: round(v, 3) for k, v in mt_trigo_amj.items()}, "p_DM_log": round(float(dm_trigo_amj["p"]), 4)})

# 3.2 Rolling h = 2 (1996-2024, n = 29) para soja y maíz
TEST_H2 = list(range(1996, 2025))
nina_h2 = (X.loc[TEST_H2, "oni_jas"] <= -0.5).to_numpy()
res_h2 = {}
rows_h2 = []
for c in ("soja", "maiz"):
    ly_all = np.log(Y[c])
    f_h2 = []
    for t in TEST_H2:
        tr = ly_all.loc[:t - 2]
        anios = tr.index
        ly = tr.reset_index(drop=True)
        orden, r2 = arima_auto(ly)
        fc2 = r2.get_forecast(2)
        p2 = float(np.exp(fc2.predicted_mean.iloc[1]))
        ic80_2 = np.exp(fc2.conf_int(alpha=0.2).iloc[1]).to_numpy()
        ic95_2 = np.exp(fc2.conf_int(alpha=0.05).iloc[1]).to_numpy()

        r3 = ARIMA(ly, exog=X.loc[anios, ["oni_jas"]].to_numpy(), order=orden, trend="t").fit()
        fc3 = r3.get_forecast(2, exog=X.loc[[t - 1, t], ["oni_jas"]].to_numpy())
        p3 = float(np.exp(fc3.predicted_mean.iloc[1]))
        ic80_3 = np.exp(fc3.conf_int(alpha=0.2).iloc[1]).to_numpy()
        ic95_3 = np.exp(fc3.conf_int(alpha=0.05).iloc[1]).to_numpy()
        obs_t = float(Y.loc[t, c])
        f_h2.append({
            "campania": t, "obs": obs_t,
            "M2_h2": p2, "M2_cov80": float(ic80_2[0] <= obs_t <= ic80_2[1]), "M2_cov95": float(ic95_2[0] <= obs_t <= ic95_2[1]),
            "M3_h2": p3, "M3_cov80": float(ic80_3[0] <= obs_t <= ic80_3[1]), "M3_cov95": float(ic95_3[0] <= obs_t <= ic95_3[1]),
        })
    df_h2 = pd.DataFrame(f_h2).set_index("campania")
    res_h2[c] = df_h2
    dm_h2_log = dm_test(np.log(df_h2.obs) - np.log(df_h2.M3_h2), np.log(df_h2.obs) - np.log(df_h2.M2_h2), h=2, alternativa="less")
    dm_h2_kg = dm_test(df_h2.obs - df_h2.M3_h2, df_h2.obs - df_h2.M2_h2, h=2, alternativa="less")
    m2_mt = resumen_metricas(df_h2.obs, df_h2.M2_h2, df_h2.M2_h2, nina_h2)
    m3_mt = resumen_metricas(df_h2.obs, df_h2.M3_h2, df_h2.M2_h2, nina_h2)
    rows_h2.append({
        "cultivo": c, "n": len(TEST_H2),
        "M2_RMSE_h2": round(m2_mt["RMSE kg/ha"], 2),
        "M3_RMSE_h2": round(m3_mt["RMSE kg/ha"], 2),
        "skill_h2_%": round(m3_mt["skill vs ARIMA %"], 2),
        "M2_cov80_%": round(100 * df_h2.M2_cov80.mean(), 1),
        "M2_cov95_%": round(100 * df_h2.M2_cov95.mean(), 1),
        "M3_cov80_%": round(100 * df_h2.M3_cov80.mean(), 1),
        "M3_cov95_%": round(100 * df_h2.M3_cov95.mean(), 1),
        "p_DM_h2_log": round(float(dm_h2_log["p"]), 4),
        "p_DM_h2_kg": round(float(dm_h2_kg["p"]), 4),
    })

print("\n3.2 Evaluación rolling h = 2 (1996–2024, n = 29):")
print(pd.DataFrame(rows_h2).to_string(index=False))

# 3.3 Puente explícito Etapa 3a -> 3b: ARIMA(2,0,2) sobre n34_anom hasta septiembre de cada año
n34 = enso["n34_anom"].dropna()
n34_djf_hat = {}
for s in range(1970, 2025):
    corte = pd.Timestamp(f"{s}-09-01")
    res_n34 = ARIMA(n34.loc[:corte], order=(2, 0, 2), trend="c").fit()
    fc_n34 = res_n34.forecast(5)  # oct(1), nov(2), dic(3), ene(4), feb(5)
    n34_djf_hat[s] = float(fc_n34.iloc[2:5].mean())
X["n34_djf_hat"] = pd.Series(n34_djf_hat)
corr_puente_test = float(X.loc[TEST, "n34_djf_hat"].corr(X.loc[TEST, "oni_djf"]))
rmse_puente_enso = float(np.sqrt(np.mean((X.loc[TEST, "n34_djf_hat"] - X.loc[TEST, "oni_djf"]) ** 2)))
print(f"\n3.3 Puente ENSO (1995–2024): corr(n34_djf_hat desde sep, oni_djf obs) = {corr_puente_test:.3f}, RMSE = {rmse_puente_enso:.3f} °C")

puente_tabla = []
for c in CULTIVOS:
    ly_all = np.log(Y[c])
    preds_puente = {}
    for t in TEST:
        tr = ly_all.loc[:t - 1]
        anios = tr.index
        ly = tr.reset_index(drop=True)
        orden = ast.literal_eval(pron_04[c].loc[t, "orden"])
        r_p = ARIMA(ly, exog=X.loc[anios, ["n34_djf_hat"]].to_numpy(), order=orden, trend="t").fit()
        preds_puente[t] = float(np.exp(r_p.forecast(1, exog=X.loc[[t], ["n34_djf_hat"]].to_numpy()).iloc[0]))
    ext_rolling[c]["M3_puente DJF_hat"] = pd.Series(preds_puente)

    p04 = pron_04[c]
    obs, m2 = p04["obs"], p04["M2 ARIMA"]
    for mdl, sr in [("M2 ARIMA", m2),
                    ("M3 (ONI JAS obs.)", p04["M3 ARIMAX (ONI jul–sep)"]),
                    ("M3_puente (Niño3.4 DJF hat)", ext_rolling[c]["M3_puente DJF_hat"]),
                    ("M4 (Oráculo DJF obs.)", p04["M4 ARIMAX oráculo (ONI dic–feb)"])]:
        mt = resumen_metricas(obs, sr, m2, nina_mask)
        p_log = np.nan if mdl == "M2 ARIMA" else float(dm_test(np.log(obs) - np.log(sr), np.log(obs) - np.log(m2), h=1, alternativa="less")["p"])
        puente_tabla.append({"cultivo": c, "modelo": mdl, **mt, "p_DM_log": p_log})

puente_df = pd.DataFrame(puente_tabla)
print("\nComparación Puente 3a → 3b:")
print(puente_df.round(3).to_string(index=False))

ext_results["bloque3_trigo_amj"] = {**mt_trigo_amj, "p_DM_log": float(dm_trigo_amj["p"])}
ext_results["bloque3_rolling_h2"] = rows_h2
ext_results["bloque3_puente_enso_skill"] = {"corr_1995_2024": corr_puente_test, "rmse_1995_2024": rmse_puente_enso}
ext_results["bloque3_puente_3a_3b"] = puente_df.round(3).to_dict("records")

# %%
# Figura 18: Desambiguación de baselines + puente 3a->3b
fig, axs = plt.subplots(1, 3, figsize=(13.4, 5.2), sharey=True)
orden_fig18 = [
    ("M0b RW + drift", "RW + drift"),
    ("M2 ARIMA", "M2 ARIMA"),
    ("M5_sin_oni Prophet", "Prophet sin ONI"),
    ("M5 Prophet + ONI", "Prophet + ONI"),
    ("Híbrido RF (solo ONI)", "Híbrido RF (solo ONI)"),
    ("Híbrido RF", "Híbrido RF (4 vars)"),
    ("Híbrido GBM (solo ONI)", "Híbrido GBM (solo ONI)"),
    ("Híbrido GBM", "Híbrido GBM (4 vars)"),
    ("M3 ARIMAX (ONI jul–sep)", "M3 ARIMAX (ONI JAS)"),
    ("M3_puente DJF_hat", "M3 Puente (N3.4 DJF hat)"),
    ("M4 ARIMAX oráculo (ONI dic–feb)", "M4 Oráculo (ONI DJF)"),
    ("M3_trigo_AMJ", "M3 ARIMAX (ONI AMJ)"),
]
y_pos = np.arange(len(orden_fig18))

for ax, c in zip(axs, CULTIVOS):
    p04, pe = pron_04[c], ext_rolling[c]
    obs = p04["obs"]
    rmse_m2 = float(np.sqrt(np.mean((obs - p04["M2 ARIMA"]) ** 2)))
    ax.axvline(rmse_m2, color=COLOR["tinta"], lw=0.9, ls=":", alpha=0.65, zorder=2)

    rmses, cols = [], []
    for key, _ in orden_fig18:
        if key == "M3_trigo_AMJ":
            if c == "trigo":
                sr = df_trigo_amj["M3_trigo_AMJ"]
                rmses.append(float(np.sqrt(np.mean((obs - sr) ** 2))))
            else:
                rmses.append(np.nan)
        else:
            sr = p04[key] if key in p04.columns else pe[key]
            rmses.append(float(np.sqrt(np.mean((obs - sr) ** 2))))

        if key in ("M3 ARIMAX (ONI jul–sep)", "M3_puente DJF_hat", "M3_trigo_AMJ"):
            cols.append(COLOR["nino"])
        elif "Híbrido" in key:
            cols.append(COLOR["serie"])
        elif "oráculo" in key:
            cols.append("#d6d4ce")
        elif key == "M2 ARIMA":
            cols.append(COLOR["tinta"])
        else:
            cols.append(COLOR["neutral"])

    max_r = float(np.nanmax(rmses))
    valid_idx = [i for i, v in enumerate(rmses) if np.isfinite(v)]
    ax.barh([y_pos[i] for i in valid_idx], [rmses[i] for i in valid_idx],
            color=[cols[i] for i in valid_idx], height=0.66, zorder=3)
    for i, v in enumerate(rmses):
        if np.isfinite(v):
            ax.text(v + max_r * 0.015, i, f"{v:.0f}", va="center", fontsize=7.8, color=COLOR["tinta2"])
    ax.set_xlim(0, max_r * 1.16)
    ax.set_title(CULTIVO_NOMBRE[c])
    ax.set_xlabel("RMSE (kg/ha), 1995/96–2024/25")

axs[0].set_yticks(y_pos, [lab for _, lab in orden_fig18], fontsize=8.5)
axs[0].invert_yaxis()

legend_handles = [
    Patch(facecolor=COLOR["tinta"], label="M2 ARIMA (referencia)"),
    Patch(facecolor=COLOR["neutral"], label="RW + drift / Prophet"),
    Patch(facecolor=COLOR["serie"], label="Híbridos ARIMA + ML"),
    Patch(facecolor=COLOR["nino"], label="ARIMAX / Puente ENSO"),
    Patch(facecolor="#d6d4ce", label="M4 Oráculo DJF"),
]
fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.86))
fig.suptitle("Desambiguación de baselines, control causal (trigo AMJ) y puente Etapa 3a → 3b",
             x=0.01, y=0.985, ha="left", va="top", fontweight="bold")
fig.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.01, 0.925),
           ncol=5, fontsize=8, frameon=False)
fig.savefig(FIG / "18_baselines_y_puente_3a_3b.png")
plt.show()

# %% [markdown]
# ## Bloque 4 — Diagnósticos de Residuos y Diebold-Mariano Unificado + Holm
#
# Completamos los diagnósticos de `04_pronostico_rindes` (ACF/PACF de `Δlog` y Ljung-Box de residuos `M2`/`M3`) y
# unificamos el test de Diebold-Mariano (con pesos de Bartlett en `utils_ts.dm_test`) en **escala `log`** y **`kg/ha`**
# aplicando formalmente la corrección de Holm (`soja` y `maiz` para H3).

# %%
diag_residuos = []
for c in CULTIVOS:
    ly = np.log(Y[c]).reset_index(drop=True)
    orden, r2 = arima_auto(ly)
    ex_col = "oni_amj" if c == "trigo" else "oni_jas"
    r3 = ARIMA(ly, exog=X.loc[Y.index, [ex_col]].to_numpy(), order=orden, trend="t").fit()
    for nombre, res in [("M2 ARIMA", r2), (f"M3 ARIMAX ({ex_col})", r3)]:
        e = res.resid.iloc[1:]
        lb = acorr_ljungbox(e, lags=[5, 10], return_df=True)
        diag_residuos.append({
            "cultivo": c, "modelo": nombre, "orden": str(orden),
            "LB_p_lag5": float(lb.lb_pvalue.iloc[0]),
            "LB_p_lag10": float(lb.lb_pvalue.iloc[1]),
        })
print("Ljung-Box sobre residuos (1970–2024):")
print(pd.DataFrame(diag_residuos).round(4).to_string(index=False))

# Tabla maestra DM (log vs kg/ha + Holm en soja y maiz)
todos_modelos = [
    "M0 Naive", "M0b RW + drift", "M1 Tendencia lineal",
    "M3 ARIMAX (ONI jul–sep)", "M3_asim ARIMAX", "M3_puente DJF_hat",
    "M4 ARIMAX oráculo (ONI dic–feb)", "M5_sin_oni Prophet", "M5 Prophet + ONI",
    "Híbrido Ridge", "Híbrido RF (solo ONI)", "Híbrido RF",
    "Híbrido GBM (solo ONI)", "Híbrido GBM", "Híbrido MLP",
]

dm_rows = []
for mdl in todos_modelos:
    fila = {"modelo": mdl}
    for c in CULTIVOS:
        p04, pe = pron_04[c], ext_rolling[c]
        obs, m2 = p04["obs"], p04["M2 ARIMA"]
        sr = pe[mdl] if mdl in pe.columns else p04[mdl]
        r_log = dm_test(np.log(obs) - np.log(sr), np.log(obs) - np.log(m2), h=1, alternativa="less")
        r_kg = dm_test(obs - sr, obs - m2, h=1, alternativa="less")
        fila[f"{c}_DM_log"] = float(r_log["DM"])
        fila[f"{c}_p_log"] = float(r_log["p"])
        fila[f"{c}_DM_kg"] = float(r_kg["DM"])
        fila[f"{c}_p_kg"] = float(r_kg["p"])
    # Holm entre los dos cultivos primarios (soja y maiz)
    holm_log = multipletests([fila["soja_p_log"], fila["maiz_p_log"]], method="holm")[1]
    holm_kg = multipletests([fila["soja_p_kg"], fila["maiz_p_kg"]], method="holm")[1]
    fila["soja_pHolm_log"], fila["maiz_pHolm_log"] = float(holm_log[0]), float(holm_log[1])
    fila["soja_pHolm_kg"], fila["maiz_pHolm_kg"] = float(holm_kg[0]), float(holm_kg[1])
    dm_rows.append(fila)

dm_master = pd.DataFrame(dm_rows)
cols_show = ["modelo", "soja_p_log", "soja_pHolm_log", "soja_p_kg", "soja_pHolm_kg",
             "maiz_p_log", "maiz_pHolm_log", "maiz_p_kg", "maiz_pHolm_kg", "trigo_p_log"]
print("\nTabla maestra Diebold-Mariano (unilateral less vs M2 ARIMA) + Holm (soja, maíz):")
print(dm_master[cols_show].round(4).to_string(index=False))

ext_results["bloque4_diagnosticos_ljungbox"] = pd.DataFrame(diag_residuos).round(4).to_dict("records")
ext_results["bloque4_dm_unificado_holm"] = dm_master.round(4).to_dict("records")

# %%
# Figura 19: ACF/PACF de Δlog + comparación p_DM (log vs kg/ha y Holm)
fig, axs = plt.subplots(2, 3, figsize=(13.2, 6.4))

for j, c in enumerate(CULTIVOS):
    ax = axs[0, j]
    dlog = np.log(Y[c]).diff().dropna()
    plot_acf(dlog, lags=10, ax=ax, color=CULTIVO_COLOR[c], vlines_kwargs={"colors": CULTIVO_COLOR[c]}, title="")
    ax.set_title(f"ACF Δlog — {CULTIVO_NOMBRE[c]} (LB M2 p={diag_residuos[2*j]['LB_p_lag5']:.2f})", fontsize=9.5)
    ax.set_xlabel("lag (campañas)", fontsize=8.5)

ax_soja, ax_maiz, ax_pacf = axs[1, 0], axs[1, 1], axs[1, 2]

# PACF de Δlog maíz (cultivo clave con quiebre de tendencia)
plot_pacf(np.log(Y["maiz"]).diff().dropna(), lags=10, method="ywm", ax=ax_pacf,
          color=CULTIVO_COLOR["maiz"], vlines_kwargs={"colors": CULTIVO_COLOR["maiz"]}, title="")
ax_pacf.set_title("PACF Δlog — Maíz (1970–2024)", fontsize=9.5)
ax_pacf.set_xlabel("lag (campañas)", fontsize=8.5)

sel_dm = ["M3 ARIMAX (ONI jul–sep)", "M3_asim ARIMAX", "M3_puente DJF_hat",
          "M5 Prophet + ONI", "Híbrido RF (solo ONI)", "Híbrido RF", "Híbrido GBM"]
short_dm = ["ARIMAX", "ARIMAX asim", "Puente 3a→3b", "Prophet+ONI", "RF solo ONI", "RF 4 vars", "GBM 4 vars"]
sub_dm = dm_master.set_index("modelo").loc[sel_dm]
y_idx = np.arange(len(sel_dm))
h_bar = 0.27
for ax, c in [(ax_soja, "soja"), (ax_maiz, "maiz")]:
    ax.barh(y_idx - h_bar, -np.log10(sub_dm[f"{c}_p_log"]), height=h_bar, color=COLOR["serie"], label="DM (log)")
    ax.barh(y_idx, -np.log10(sub_dm[f"{c}_p_kg"]), height=h_bar, color=COLOR["tendencia"], label="DM (kg/ha)")
    ax.barh(y_idx + h_bar, -np.log10(sub_dm[f"{c}_pHolm_log"]), height=h_bar, color=COLOR["tinta2"], label="Holm (log)")
    ax.axvline(-np.log10(0.05), color=COLOR["nino"], lw=1, ls="--")
    ax.set_xlim(0, max(1.62, ax.get_xlim()[1]))
    ax.text(-np.log10(0.05) + 0.04, 4.0, "p=0,05", va="center", fontsize=7.5, color=COLOR["nino"],
            bbox=dict(facecolor=COLOR["fondo"], alpha=0.85, edgecolor="none", pad=1))
    ax.set_yticks(y_idx, short_dm if c == "soja" else [""] * len(short_dm), fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("−log10(p_DM)", fontsize=8.5)
    ax.set_title(f"Diebold-Mariano vs ARIMA — {CULTIVO_NOMBRE[c]}", fontsize=9.5)
ax_soja.legend(fontsize=7.2, loc="lower right")
fig.suptitle("Diagnósticos de Δlog(rend) y sensibilidad del test Diebold-Mariano (log vs kg/ha y Holm)",
             x=0.01, ha="left", fontweight="bold")
fig.savefig(FIG / "19_diagnosticos_y_dm.png")
plt.show()

# %% [markdown]
# ## Bloque 5 — Coherencia Wavelet (`Niño 3.4` ↔ `precip_z` pampeana, `1981–2025`, `n = 540`)
#
# Implementación según **Grinsted, Moore & Jevrejeva (2004)** con wavelet Morlet (`ω₀ = 6`):
# - Suavizado gaussiano en el tiempo (escala-dependiente) y rectángulo (*boxcar*) de ancho `0,60` octavas en escala.
# - Cono de influencia (COI), flechas de fase donde `R² > 0,5` y significancia al 95 % por **250 simulaciones Monte Carlo AR(1)**.

# %%
m_wc = pd.concat([enso["n34_anom"], clima["precip_z"]], axis=1).loc["1981":"2025"].dropna()
xw = (m_wc["n34_anom"].to_numpy(float) - m_wc["n34_anom"].mean()) / m_wc["n34_anom"].std()
yw = (m_wc["precip_z"].to_numpy(float) - m_wc["precip_z"].mean()) / m_wc["precip_z"].std()

N_w = len(xw)
dt_w = 1.0 / 12.0
omega0 = 6.0
flambda = (4.0 * np.pi) / (omega0 + np.sqrt(2.0 + omega0 ** 2))  # ~1.0330 para omega0=6
dj = 1.0 / 12.0
s0 = 2.0 * dt_w
J = int(np.floor(np.log2(N_w * dt_w / s0) / dj))
scales = s0 * 2.0 ** (np.arange(J + 1) * dj)
periods = flambda * scales

# Grilla de frecuencias FFT (zero-padding a potencia de 2)
n_pad = int(2 ** np.ceil(np.log2(N_w * 2)))
k_fft = np.arange(n_pad)
omega = np.where(k_fft <= n_pad // 2, 2.0 * np.pi * k_fft / (n_pad * dt_w),
                 -2.0 * np.pi * (n_pad - k_fft) / (n_pad * dt_w))

# Matriz de wavelets en Fourier (S, n_pad), normalizada por escala según Torrence & Compo (1998)
so = np.outer(scales, omega)
heaviside = (omega > 0.0)[None, :]
norm_w = np.sqrt(2.0 * np.pi * scales / dt_w)[:, None] * (np.pi ** -0.25)
psi_hat = norm_w * np.exp(-0.5 * (so - omega0) ** 2) * heaviside  # real; conj = ella misma

# Precomputar kernels de suavizado en tiempo (Gaussiano en frecuencia o convolución directa) y escala (boxcar 0.60 octavas)
time_kernels = []
for sc in scales:
    sig_samp = sc / dt_w
    half = min(int(np.ceil(3.0 * sig_samp)), N_w)
    tt = np.arange(-half, half + 1)
    kern = np.exp(-0.5 * (tt / sig_samp) ** 2)
    kern /= kern.sum()
    time_kernels.append(kern)

n_sc_box = int(round(0.60 / dj))  # ~7 escalas para dj=1/12
if n_sc_box % 2 == 0:
    n_sc_box += 1
scale_kern = np.ones(n_sc_box) / n_sc_box


def cwt_morlet(arr: np.ndarray) -> np.ndarray:
    """CWT Morlet (ω0=6) para una serie (N,) o lote (B, N)."""
    a2d = np.atleast_2d(arr)
    f_arr = np.fft.fft(a2d, n=n_pad, axis=-1)
    prod = f_arr[:, None, :] * psi_hat[None, :, :]
    W = np.fft.ifft(prod, axis=-1)[..., :N_w]
    return W[0] if arr.ndim == 1 else W


def smooth_wt(W: np.ndarray) -> np.ndarray:
    """Suavizado tiempo + escala para (..., S, N_w)."""
    orig_2d = W.ndim == 2
    W3 = W[None, ...] if orig_2d else W
    B, S, T = W3.shape
    out_t = np.empty_like(W3)
    for j in range(S):
        k = time_kernels[j]
        half = (len(k) - 1) // 2
        for b in range(B):
            out_t[b, j] = np.convolve(W3[b, j], k, mode="full")[half:half + T]
    # Suavizado en escala (boxcar simétrico normalizado)
    pad_s = n_sc_box // 2
    out_s = np.empty_like(out_t)
    for j in range(S):
        j0, j1 = max(0, j - pad_s), min(S, j + pad_s + 1)
        out_s[:, j, :] = out_t[:, j0:j1, :].mean(axis=1)
    return out_s[0] if orig_2d else out_s


def wavelet_coherence(x_arr: np.ndarray, y_arr: np.ndarray):
    Wx = cwt_morlet(x_arr)
    Wy = cwt_morlet(y_arr)
    inv_s = (1.0 / scales)[..., None]
    Sxy = smooth_wt(inv_s * (Wx * np.conj(Wy)))
    Sxx = np.real(smooth_wt(inv_s * (np.abs(Wx) ** 2)))
    Syy = np.real(smooth_wt(inv_s * (np.abs(Wy) ** 2)))
    R2 = (np.abs(Sxy) ** 2) / np.maximum(Sxx * Syy, 1e-30)
    phase = np.angle(Sxy)
    return R2, phase


R2_obs, phase_obs = wavelet_coherence(xw, yw)
assert np.all((R2_obs >= -1e-9) & (R2_obs <= 1.0 + 1e-6)), "R² fuera de [0, 1]!"
assert float(R2_obs.mean()) < 0.99, "R² degenerado cerca de 1!"
R2_obs = np.clip(R2_obs, 0.0, 1.0)

# Significancia Monte Carlo AR(1) (250 pares independientes vectorizados en lotes)
t_mc = time.time()
phi_x = float(m_wc["n34_anom"].autocorr(1))
phi_y = float(m_wc["precip_z"].autocorr(1))
n_mc, batch_mc, burn = 250, 25, 100
rng_mc = np.random.default_rng(0)


def sim_ar1(phi: float, b_size: int) -> np.ndarray:
    eps = rng_mc.standard_normal((b_size, N_w + burn))
    out = np.empty_like(eps)
    out[:, 0] = eps[:, 0] / np.sqrt(max(1.0 - phi ** 2, 1e-6))
    for k in range(1, N_w + burn):
        out[:, k] = phi * out[:, k - 1] + eps[:, k]
    z_out = out[:, burn:]
    return (z_out - z_out.mean(axis=1, keepdims=True)) / z_out.std(axis=1, keepdims=True)


R2_mc_list = []
for b0 in range(0, n_mc, batch_mc):
    bs = min(batch_mc, n_mc - b0)
    x_mc = sim_ar1(phi_x, bs)
    y_mc = sim_ar1(phi_y, bs)
    r2_b, _ = wavelet_coherence(x_mc, y_mc)
    R2_mc_list.append(r2_b)
R2_mc = np.concatenate(R2_mc_list, axis=0)
# Umbral 95 % por escala (evaluado en el interior lejos de los bordes)
sig95_scale = np.quantile(R2_mc[:, :, 60:-60], 0.95, axis=(0, 2))
sig95_ratio = R2_obs / sig95_scale[:, None]
print(f"Coherencia wavelet + MC AR(1) ({n_mc} sims) lista en {time.time() - t_mc:.1f} s · "
      f"R² min={R2_obs.min():.3f}, mean={R2_obs.mean():.3f}, max={R2_obs.max():.3f}", flush=True)

# COI e indicadores por década dentro de COI en la banda ENSO (2–7 años)
t_dist = (0.5 + np.minimum(np.arange(N_w), N_w - 1 - np.arange(N_w))) * dt_w
coi_period = flambda * (t_dist / np.sqrt(2.0))
inside_coi = periods[:, None] <= coi_period[None, :]
band_enso = (periods >= 2.0) & (periods <= 7.0)

decadas = [("1980s (1981–89)", 1981, 1989), ("1990s (1990–99)", 1990, 1999),
           ("2000s (2000–09)", 2000, 2009), ("2010s–2020s (2010–25)", 2010, 2025)]
coh_decadas = []
years_w = m_wc.index.year.to_numpy()
for nombre_d, y0, y1 in decadas:
    mask_t = (years_w >= y0) & (years_w <= y1)
    sub_mask = band_enso[:, None] & inside_coi & mask_t[None, :]
    r2_sub = R2_obs[sub_mask]
    sig_sub = (sig95_ratio >= 1.0)[sub_mask]
    coh_decadas.append({
        "decada": nombre_d,
        "n_celdas_COI": int(sub_mask.sum()),
        "R2_media_2_7_anios": float(r2_sub.mean()),
        "R2_max_2_7_anios": float(r2_sub.max()),
        "frac_sig95_%": float(100.0 * sig_sub.mean()),
    })
print("\nCoherencia wavelet media (2–7 años, dentro del COI) por década:")
print(pd.DataFrame(coh_decadas).round(3).to_string(index=False))

ext_results["bloque5_coherencia_wavelet"] = {
    "n_meses": N_w,
    "ar1_nino34": phi_x,
    "ar1_precip_z": phi_y,
    "n_mc": n_mc,
    "R2_global_mean": float(R2_obs.mean()),
    "R2_global_max": float(R2_obs.max()),
    "decadas": coh_decadas,
}

# %%
# Figura 20: Mapa tiempo-período de coherencia wavelet
anios_ax = m_wc.index.year + (m_wc.index.month - 0.5) / 12.0
fig, ax = plt.subplots(figsize=(11.5, 4.8))
im = ax.contourf(anios_ax, periods, R2_obs, levels=np.linspace(0, 1, 21), cmap="YlGnBu")
ax.contour(anios_ax, periods, sig95_ratio, levels=[1.0], colors=[COLOR["tinta"]], linewidths=1.3)
ax.fill_between(anios_ax, coi_period, periods.max(), color=COLOR["fondo"], alpha=0.68, lw=0)
ax.plot(anios_ax, coi_period, color=COLOR["tinta2"], lw=1.0, ls="--")

# Flechas de fase donde R² > 0.5 y dentro de COI
step_t, step_s = 18, 4
tt_idx = np.arange(8, N_w, step_t)
ss_idx = np.arange(4, len(periods) - 2, step_s)
TT, SS = np.meshgrid(tt_idx, ss_idx)
mask_arr = (R2_obs[SS, TT] > 0.5) & (periods[SS] <= coi_period[TT])
U = np.cos(phase_obs[SS, TT]) * mask_arr
V = np.sin(phase_obs[SS, TT]) * mask_arr
if mask_arr.any():
    ax.quiver(anios_ax[TT[mask_arr]], periods[SS[mask_arr]], U[mask_arr], V[mask_arr],
              units="inches", scale=10, width=0.015, headwidth=3.5, headlength=4.0,
              color=COLOR["tinta"], pivot="mid", zorder=4)

for b_p in (2.0, 7.0):
    ax.axhline(b_p, color=COLOR["nino"], lw=0.9, ls=":", alpha=0.8)
ax.set_yscale("log")
ax.set_yticks([0.25, 0.5, 1, 2, 4, 8, 16], ["0,25", "0,5", "1", "2", "4", "8", "16"])
ax.set_ylim(min(16.0, periods.max()), periods.min())
ax.set_ylabel("período (años)")
ax.set_xlabel("año")
ax.set_title("Coherencia wavelet al cuadrado Niño 3.4 ↔ lluvia pampeana (1981–2025) · contorno: AR(1) 95 % · flechas → en fase")
ax.grid(False)
cb = fig.colorbar(im, ax=ax, pad=0.01)
cb.set_label("coherencia R²")
fig.savefig(FIG / "20_coherencia_wavelet.png")
plt.show()

# %% [markdown]
# ## Exportación de resultados consolidados (`results/etapa4_extensiones.json` y CSV)

# %%
df_rolling_ext = pd.concat([df.assign(cultivo=c).reset_index() for c, df in ext_rolling.items()], ignore_index=True)
df_rolling_ext = df_rolling_ext.merge(df_trigo_amj.reset_index().assign(cultivo="trigo"), on=["cultivo", "campania"], how="left")
df_rolling_ext.to_csv(RES / "pronosticos_rolling_extensiones.csv", index=False)
(RES / "etapa4_extensiones.json").write_text(json.dumps(ext_results, indent=2, default=float, ensure_ascii=False), encoding="utf-8")
print("Guardado results/etapa4_extensiones.json y results/pronosticos_rolling_extensiones.csv", flush=True)
