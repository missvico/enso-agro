"""Modelado en ventana móvil expansiva (sin fuga futura) y puente Etapa 3a → 3b."""
from __future__ import annotations

import ast
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from statsmodels.tools.sm_exceptions import ConvergenceWarning, EstimationWarning
from statsmodels.tsa.arima.model import ARIMA


def ajustar_arima(
    ly: pd.Series,
    order: tuple[int, int, int],
    exog=None,
    trend: str = "t",
    conv_log: list[dict] | None = None,
    contexto: str = "",
    cultivo: str = "",
    anio: int | None = None,
):
    """Ajusta ARIMA/ARIMAX capturando ConvergenceWarning y parámetros MA cerca de |θ|=1."""
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always", ConvergenceWarning)
        warnings.filterwarnings("ignore", category=EstimationWarning)
        res = ARIMA(ly, exog=exog, order=order, trend=trend).fit()
    cws = [w for w in ws if issubclass(w.category, ConvergenceWarning)]
    for w in ws:
        if not issubclass(w.category, ConvergenceWarning):
            warnings.warn(w.message, w.category)
    if conv_log is not None:
        ma_vals = {k: round(float(v), 5) for k, v in res.params.items() if k.startswith("ma.")}
        cerca_borde = any(abs(abs(v) - 1.0) < 0.05 for v in ma_vals.values())
        if cws or cerca_borde:
            conv_log.append({
                "contexto": contexto,
                "cultivo": cultivo,
                "campania": anio,
                "orden": str(order),
                "conv_warning": bool(cws),
                "cerca_borde_MA": cerca_borde,
                "ma_params": ma_vals,
                "mensaje": str(cws[0].message) if cws else "",
            })
    return res


def aicc(res) -> float:
    k = len(res.params)
    return float(res.aic + 2 * k * (k + 1) / max(res.nobs - k - 1, 1))


def arima_auto(ly: pd.Series, exog=None):
    """auto.arima reducido idéntico a Etapa 3b: (p,1,q) con drift, p,q ∈ {0,1,2}."""
    mejor = None
    for p in range(3):
        for q in range(3):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    warnings.simplefilter("ignore", EstimationWarning)
                    r = ARIMA(ly, exog=exog, order=(p, 1, q), trend="t").fit()
            except Exception:
                continue
            if mejor is None or aicc(r) < aicc(mejor[1]):
                mejor = ((p, 1, q), r)
    return mejor


def rolling_extensiones_h1(
    Y: pd.DataFrame,
    X: pd.DataFrame,
    pron_04: dict[str, pd.DataFrame],
    cultivos: list[str] | tuple[str, ...],
    test_years: list[int],
    include_prophet: bool = True,
    conv_log: list[dict] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """Origen móvil causal h = 1 (solo años <= t-1 en entrenamiento) para baselines desambiguados y M3_asim."""
    ext_rolling: dict[str, pd.DataFrame] = {}
    m2_reajuste_stats: dict[str, dict] = {}

    for c in cultivos:
        ly_all = np.log(Y[c])
        ex_col = "oni_amj" if c == "trigo" else "oni_jas"
        filas = []
        max_rel_diff = 0.0
        tol_m2 = 2e-3 if c != "trigo" else 1.5e-2

        for t in test_years:
            # Corte estricto: entrenamiento usa únicamente campañas <= t - 1 (< t)
            tr = ly_all.loc[:t - 1]
            anios = tr.index
            ly = tr.reset_index(drop=True)
            orden = ast.literal_eval(pron_04[c].loc[t, "orden"])
            m2_ref = float(pron_04[c].loc[t, "M2 ARIMA"])

            r2 = ajustar_arima(
                ly, order=orden, trend="t", conv_log=conv_log,
                contexto="B1_M2_resid", cultivo=c, anio=t,
            )
            pred_m2_local = float(np.exp(r2.forecast(1).iloc[0]))
            rel_diff = abs(pred_m2_local - m2_ref) / m2_ref
            max_rel_diff = max(max_rel_diff, rel_diff)
            assert np.isclose(pred_m2_local, m2_ref, rtol=tol_m2), (
                f"{c} {t}: pred_m2={pred_m2_local:.4f} vs pron_04={m2_ref:.4f} (rel={rel_diff:.4e})"
            )

            drift = float(ly.diff().dropna().mean())
            pred_rw_drift = float(np.exp(ly.iloc[-1] + drift))

            if include_prophet:
                from prophet import Prophet

                dfp = pd.DataFrame({"ds": pd.to_datetime(anios.astype(str)), "y": ly.to_numpy()})
                mp = Prophet(
                    yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
                    changepoint_prior_scale=0.05, uncertainty_samples=0,
                )
                mp.fit(dfp)
                pred_prophet_sin = float(np.exp(mp.predict(pd.DataFrame({"ds": [pd.Timestamp(f"{t}")]}))["yhat"].iloc[0]))
            else:
                pred_prophet_sin = float("nan")

            # Híbridos RF y GBM solo con ONI causal (promedio 5 semillas 0..4; trigo usa oni_amj)
            resid = r2.resid.iloc[1:].to_numpy()
            X_oni_tr = X.loc[anios[1:], [ex_col]].to_numpy()
            X_oni_te = X.loc[[t], [ex_col]].to_numpy()
            rf_preds = [
                RandomForestRegressor(n_estimators=400, max_depth=3, min_samples_leaf=4, random_state=s)
                .fit(X_oni_tr, resid).predict(X_oni_te)[0]
                for s in range(5)
            ]
            gbm_preds = [
                GradientBoostingRegressor(
                    n_estimators=150, max_depth=2, learning_rate=0.03,
                    subsample=0.8, min_samples_leaf=4, random_state=s,
                ).fit(X_oni_tr, resid).predict(X_oni_te)[0]
                for s in range(5)
            ]

            # M3_asim: ARIMAX asimétrico (Bloque 2 rolling; trigo usa oni_amj)
            oni_tr_arr = X.loc[anios, ex_col].to_numpy()
            ex_asim_tr = np.column_stack([np.maximum(oni_tr_arr, 0.0), np.minimum(oni_tr_arr, 0.0)])
            oni_t_val = float(X.loc[t, ex_col])
            ex_asim_te = np.array([[max(oni_t_val, 0.0), min(oni_t_val, 0.0)]])
            r_asim = ajustar_arima(
                ly, order=orden, exog=ex_asim_tr, trend="t", conv_log=conv_log,
                contexto="B2_M3_asim", cultivo=c, anio=t,
            )
            pred_asim = float(np.exp(r_asim.forecast(1, exog=ex_asim_te).iloc[0]))

            filas.append({
                "cultivo": c,
                "campania": t,
                "obs": float(Y.loc[t, c]),
                "M2 ARIMA": m2_ref,
                "M0b RW + drift": pred_rw_drift,
                "M5_sin_oni Prophet": pred_prophet_sin,
                "Híbrido RF (solo ONI)": float(m2_ref * np.exp(np.mean(rf_preds))),
                "Híbrido GBM (solo ONI)": float(m2_ref * np.exp(np.mean(gbm_preds))),
                "M3_asim ARIMAX": pred_asim,
            })

        ext_rolling[c] = pd.DataFrame(filas).set_index("campania")
        m2_reajuste_stats[c] = {"max_rel_diff": round(max_rel_diff, 6), "tol": tol_m2}

    return ext_rolling, m2_reajuste_stats


def rolling_trigo_amj(
    y_trigo: pd.Series,
    oni_amj: pd.Series,
    pron_04_trigo: pd.DataFrame,
    test_years: list[int],
    conv_log: list[dict] | None = None,
) -> pd.DataFrame:
    """ARIMAX causal en trigo usando ONI AMJ (previo a siembra jun–jul)."""
    ly_trigo = np.log(y_trigo)
    filas = []
    for t in test_years:
        tr = ly_trigo.loc[:t - 1]
        anios = tr.index
        ly = tr.reset_index(drop=True)
        orden = ast.literal_eval(pron_04_trigo.loc[t, "orden"])
        r_amj = ajustar_arima(
            ly, order=orden, exog=oni_amj.loc[anios].to_numpy()[:, None], trend="t",
            conv_log=conv_log, contexto="B3_trigo_AMJ", cultivo="trigo", anio=t,
        )
        pred_amj = float(np.exp(r_amj.forecast(1, exog=oni_amj.loc[[t]].to_numpy()[:, None]).iloc[0]))
        filas.append({"campania": t, "M3_trigo_AMJ": pred_amj})
    return pd.DataFrame(filas).set_index("campania")


def rolling_arimax_h2(
    y_cultivo: pd.Series,
    oni_jas: pd.Series,
    test_years: list[int],
    cultivo: str = "",
    conv_log: list[dict] | None = None,
) -> pd.DataFrame:
    """Horizonte h = 2 en ventana móvil: entrena estrictamente hasta t - 2 (< t - 1) y pronostica t."""
    ly_all = np.log(y_cultivo)
    filas = []
    for t in test_years:
        tr = ly_all.loc[:t - 2]
        anios = tr.index
        ly = tr.reset_index(drop=True)
        orden, r2 = arima_auto(ly)
        fc2 = r2.get_forecast(2)
        p2 = float(np.exp(fc2.predicted_mean.iloc[1]))
        ic80_2 = np.exp(fc2.conf_int(alpha=0.2).iloc[1]).to_numpy()
        ic95_2 = np.exp(fc2.conf_int(alpha=0.05).iloc[1]).to_numpy()

        r3 = ajustar_arima(
            ly, order=orden, exog=oni_jas.loc[anios].to_numpy()[:, None], trend="t",
            conv_log=conv_log, contexto="B3_h2_M3", cultivo=cultivo, anio=t,
        )
        fc3 = r3.get_forecast(2, exog=oni_jas.loc[[t - 1, t]].to_numpy()[:, None])
        p3 = float(np.exp(fc3.predicted_mean.iloc[1]))
        ic80_3 = np.exp(fc3.conf_int(alpha=0.2).iloc[1]).to_numpy()
        ic95_3 = np.exp(fc3.conf_int(alpha=0.05).iloc[1]).to_numpy()
        obs_t = float(y_cultivo.loc[t])
        filas.append({
            "campania": t,
            "obs": obs_t,
            "M2_h2": p2,
            "M2_cov80": float(ic80_2[0] <= obs_t <= ic80_2[1]),
            "M2_cov95": float(ic95_2[0] <= obs_t <= ic95_2[1]),
            "M3_h2": p3,
            "M3_cov80": float(ic80_3[0] <= obs_t <= ic80_3[1]),
            "M3_cov95": float(ic95_3[0] <= obs_t <= ic95_3[1]),
        })
    return pd.DataFrame(filas).set_index("campania")


def pronosticar_puente_n34_djf(
    n34_anom: pd.Series,
    years: range | list[int],
    conv_log: list[dict] | None = None,
) -> pd.Series:
    """Puente Etapa 3a: para cada s, ajusta ARIMA(2,0,2) sobre Niño 3.4 hasta <= s-09-01 y pronostica DJF."""
    n34 = n34_anom.dropna().asfreq("MS")
    n34_djf_hat = {}
    for s in years:
        corte = pd.Timestamp(f"{s}-09-01")
        sub = n34.loc[:corte]
        if sub.isna().any():
            sub = n34_anom.dropna().loc[:corte]
        res_n34 = ajustar_arima(
            sub, order=(2, 0, 2), trend="c", conv_log=conv_log,
            contexto="B3_puente_n34", cultivo="enso", anio=s,
        )
        fc_n34 = res_n34.forecast(5)  # oct(1), nov(2), dic(3), ene(4), feb(5)
        n34_djf_hat[s] = float(fc_n34.iloc[2:5].mean())
    return pd.Series(n34_djf_hat)


def rolling_puente_rindes(
    y_cultivo: pd.Series,
    n34_djf_hat: pd.Series,
    pron_04_c: pd.DataFrame,
    test_years: list[int],
    cultivo: str = "",
    conv_log: list[dict] | None = None,
) -> pd.Series:
    """Puente Etapa 3b: ARIMAX móvil usando Niño 3.4 DJF pronosticado en septiembre."""
    ly_all = np.log(y_cultivo)
    preds_puente = {}
    for t in test_years:
        tr = ly_all.loc[:t - 1]
        anios = tr.index
        ly = tr.reset_index(drop=True)
        orden = ast.literal_eval(pron_04_c.loc[t, "orden"])
        r_p = ajustar_arima(
            ly, order=orden, exog=n34_djf_hat.loc[anios].to_numpy()[:, None], trend="t",
            conv_log=conv_log, contexto="B3_puente_arimax", cultivo=cultivo, anio=t,
        )
        preds_puente[t] = float(np.exp(r_p.forecast(1, exog=n34_djf_hat.loc[[t]].to_numpy()[:, None]).iloc[0]))
    return pd.Series(preds_puente)
