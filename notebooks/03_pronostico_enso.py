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
# # Etapa 3a — Pronóstico mensual del Niño 3.4: ARIMA (Clase 11) contra redes neuronales y Prophet (Clase 12)
#
# **¿Por qué pronosticar el ENSO?** El ONI de jul–sep ya predice el rinde (H1, Etapa 2). Pero el ENSO de
# dic–feb, cuando el cultivo está en el período crítico, explica todavía más. Si en la siembra (sep–oct) se pudiera pronosticar bien
# el Niño 3.4 de los meses siguientes, ese pronóstico sería un insumo para el modelo de rindes.
#
# Acá **sí hay datos** para los modelos de la Clase 12: 918 meses. El diseño es este:
# - **Entrenamiento:** 1950–1999. **Test:** orígenes mensuales de 2000-01 a 2026-06 (origen móvil), con horizontes de 1 a 9 meses.
# - Modelos:
#   1. Persistencia (último valor).
#   2. Climatología (0).
#   3. ARIMA elegido por ACF/PACF y AICc, reajustado cada enero con ventana expanding.
#   4. MLP.
#   5. CNN 1D.
#   6. LSTM.
#   7. Prophet.
# - Métricas por horizonte: RMSE y correlación de anomalías (ACC). Usamos Diebold-Mariano para comparar con ARIMA.

# %%
import json
import os
import sys
import time
import warnings
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.neural_network import MLPRegressor
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL

RAIZ = Path.cwd().resolve()
if RAIZ.name == "notebooks":
    RAIZ = RAIZ.parent
sys.path.insert(0, str(RAIZ / "src"))
from utils_ts import COLOR, adf_kpss, dm_test, estilo, metricas

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)
estilo()
torch.set_num_threads(4)
PROC, FIG, RES = RAIZ / "data" / "processed", RAIZ / os.environ.get("FIG_DIR", "figures"), RAIZ / "results"
FIG.mkdir(exist_ok=True, parents=True)
enso = pd.read_parquet(PROC / "enso_mensual.parquet")
x = enso["n34_anom"].dropna()
H, L = 9, 24                                   # horizonte máximo y ventana de entrada (meses)
FIN_TRAIN = pd.Timestamp("1999-12-01")
ORIGENES = x.loc["2000-01":].index
print(f"Niño 3.4: {x.index[0]:%Y-%m} a {x.index[-1]:%Y-%m} ({len(x)} meses) · orígenes de test: {len(ORIGENES)}")
MODELO_COLOR = {"Persistencia": COLOR["neutral"], "Climatología": "#d6d4ce", "ARIMA": COLOR["tinta"],
                "MLP": "#1baf7a", "CNN 1D": "#6250d6", "LSTM": COLOR["serie"], "Prophet": COLOR["tendencia"],
                "ARIMA + MLP": COLOR["nino"]}

# %% [markdown]
# ## 1. Descomposición STL de la temperatura del Niño 3.4 (Clase 11)
#
# Con la temperatura **absoluta** (no la anomalía) se ve el ciclo anual y una tendencia de calentamiento. La STL
# (loess, `robust=True`) separa tendencia, estacionalidad y resto, y el resto es la señal ENSO. NOAA hace esto
# mismo cuando resta una climatología móvil de 30 años: sin ese paso, el calentamiento de fondo se confundiría con El Niño.

# %%
sst = enso["n34"].dropna()
stl = STL(sst, period=12, trend=241, robust=True).fit()
fig, axs = plt.subplots(4, 1, figsize=(11, 7), sharex=True)
for ax, serie, nombre in zip(axs, [sst, stl.trend, stl.seasonal, stl.resid],
                             ["observado (°C)", "tendencia", "estacional", "resto"]):
    ax.plot(serie.index, serie, color=COLOR["tinta2"] if nombre != "tendencia" else COLOR["tendencia"], lw=0.9)
    ax.set_ylabel(nombre)
axs[0].set_title("STL de la TSM Niño 3.4 (ventana de tendencia 20 años, estacionalidad periódica)")
fig.savefig(FIG / "09_stl_nino34.png")
plt.show()
cal = stl.trend.iloc[-12:].mean() - stl.trend.iloc[:12].mean()
print(f"calentamiento de la tendencia STL 1950→2026: {cal:+.2f} °C")
print(f"corr(resto STL, anomalía oficial) = {stl.resid.corr(enso['n34_anom']):.3f}")
print(pd.DataFrame({"TSM cruda": adf_kpss(sst), "resto STL": adf_kpss(stl.resid), "anomalía": adf_kpss(x)}).T.to_string())

# %% [markdown]
# ## 2. Identificación ARIMA por ACF/PACF y criterios de información (Clase 11)
#
# Según la tabla de la clase, una ACF que decae gradualmente y una PACF que se corta en p indican un AR(p).

# %%
xtr = x.loc[:FIN_TRAIN]
fig, axs = plt.subplots(1, 2, figsize=(12, 3.6))
plot_acf(xtr, lags=36, ax=axs[0], color=COLOR["serie"], vlines_kwargs={"colors": COLOR["serie"]})
plot_pacf(xtr, lags=36, ax=axs[1], method="ywm", color=COLOR["serie"], vlines_kwargs={"colors": COLOR["serie"]})
axs[0].set_title("ACF Niño 3.4 (1950–1999)")
axs[1].set_title("PACF Niño 3.4 (1950–1999)")
fig.savefig(FIG / "10_acf_pacf_nino34.png")
plt.show()

# %%
grilla = []
for p in range(0, 5):
    for q in range(0, 4):
        if p == q == 0:
            continue
        try:
            r = ARIMA(xtr, order=(p, 0, q), trend="c").fit()
            k = len(r.params)
            grilla.append({"orden": f"({p},0,{q})", "p": p, "q": q, "AIC": r.aic, "AICc": r.aic + 2 * k * (k + 1) / (len(xtr) - k - 1),
                           "BIC": r.bic})
        except Exception:
            pass
grilla = pd.DataFrame(grilla).sort_values("AICc")
print(grilla.head(8).round(1).to_string(index=False))
mejor = grilla.iloc[0]
ORDEN = (int(mejor.p), 0, int(mejor.q))
print(f"\nelegido por AICc: ARIMA{ORDEN}")

# %%
res_tr = ARIMA(xtr, order=ORDEN, trend="c").fit()
print(res_tr.summary().tables[1])
e = res_tr.resid.iloc[5:]
lb = acorr_ljungbox(e, lags=[12, 24], return_df=True)
print(f"Ljung-Box: p(12) = {lb.lb_pvalue.iloc[0]:.3f}, p(24) = {lb.lb_pvalue.iloc[1]:.3f} · Durbin-Watson = {durbin_watson(e):.2f}")
var_mes = e.groupby(e.index.month).std()
print("desvío de los residuos por mes calendario (heterocedasticidad estacional):")
print(var_mes.round(3).to_string())

fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
axs[0].plot(e.index, e, color=COLOR["tinta2"], lw=0.6)
axs[0].set_title("residuos ARIMA")
plot_acf(e, lags=24, ax=axs[1], color=COLOR["serie"], vlines_kwargs={"colors": COLOR["serie"]})
axs[1].set_title("ACF de residuos")
axs[2].hist(e, bins=40, density=True, color=COLOR["serie"], edgecolor=COLOR["fondo"])
xx = np.linspace(e.min(), e.max(), 200)
axs[2].plot(xx, stats.norm.pdf(xx, 0, e.std()), color=COLOR["tendencia"], lw=1.5)
axs[2].set_title(f"histograma (curtosis exceso = {stats.kurtosis(e):.1f})")
fig.savefig(FIG / "11_residuos_arima.png")
plt.show()

# %% [markdown]
# ## 3. Pronósticos con origen móvil
#
# Para cada origen t (fin de mes, de 2000-01 en adelante), cada modelo usa solo los datos hasta t y pronostica t+1 a t+9.

# %%
idx = x.index
pos = {d: i for i, d in enumerate(idx)}
Yobs = pd.DataFrame({h: x.shift(-h) for h in range(1, H + 1)}).loc[ORIGENES]   # objetivo de cada origen
pron = {}

pron["Persistencia"] = pd.DataFrame({h: x.loc[ORIGENES] for h in range(1, H + 1)})
pron["Climatología"] = pd.DataFrame(0.0, index=ORIGENES, columns=range(1, H + 1))

t0 = time.time()
filas = {}
for anio in sorted(set(ORIGENES.year)):
    res = ARIMA(x.loc[:f"{anio - 1}-12"], order=ORDEN, trend="c").fit()        # reajuste anual (expanding)
    for o in ORIGENES[ORIGENES.year == anio]:
        filas[o] = res.apply(x.loc[:o]).forecast(H).to_numpy()
pron["ARIMA"] = pd.DataFrame.from_dict(filas, orient="index", columns=range(1, H + 1))
print(f"ARIMA listo en {time.time() - t0:.0f} s")

# %% [markdown]
# ### Redes neuronales (Clase 12)
#
# Usamos una estrategia **directa multi-horizonte**: a partir de los últimos 24 meses (más el mes calendario en seno
# y coseno, porque el ENSO está "anclado" a las estaciones) la red predice los 9 meses siguientes de una vez.
# - Entrenamiento con orígenes 1950–1989. *Early stopping* con validación en 1990–1999.
# - Promedio de un ensamble de 5 semillas, para que el resultado no dependa del azar de la inicialización.
# - Los pesos quedan **fijos** en el test (como en la clase: train → test). No se reentrena con datos de 2000 en adelante.

# %%
mu, sd = xtr.mean(), xtr.std()
z = ((x - mu) / sd).to_numpy(np.float32)
mes = idx.month.to_numpy()
seno, coseno = np.sin(2 * np.pi * mes / 12).astype(np.float32), np.cos(2 * np.pi * mes / 12).astype(np.float32)


def ventanas(origenes):
    X, Yv, ok = [], [], []
    for o in origenes:
        i = pos[o]
        if i < L - 1:
            continue
        sl = slice(i - L + 1, i + 1)
        X.append(np.stack([z[sl], seno[sl], coseno[sl]], axis=1))
        y = z[i + 1:i + 1 + H]
        Yv.append(np.pad(y, (0, H - len(y)), constant_values=np.nan))
        ok.append(o)
    return np.array(X), np.array(Yv), pd.DatetimeIndex(ok)


o_fit = idx[(idx <= pd.Timestamp("1989-12-01") - pd.DateOffset(months=H))]
o_val = idx[(idx >= "1990-01-01") & (idx <= FIN_TRAIN - pd.DateOffset(months=H))]
Xf, Yf, _ = ventanas(o_fit)
Xv, Yv, _ = ventanas(o_val)
Xt, _, o_test = ventanas(ORIGENES)
print(f"muestras: entrenamiento {len(Xf)}, validación {len(Xv)}, test {len(Xt)}")


class CNN1D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(3, 16, 5, padding=2), torch.nn.ReLU(),
            torch.nn.Conv1d(16, 16, 5, padding=2), torch.nn.ReLU(), torch.nn.MaxPool1d(2))
        self.head = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Dropout(0.2), torch.nn.Linear(16 * L // 2, H))

    def forward(self, xb):                      # xb: (lote, tiempo, canales)
        return self.head(self.conv(xb.transpose(1, 2)))


class LSTMNet(torch.nn.Module):
    def __init__(self, oculto=32):
        super().__init__()
        self.lstm = torch.nn.LSTM(3, oculto, batch_first=True)
        self.head = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(oculto, H))

    def forward(self, xb):
        salida, _ = self.lstm(xb)
        return self.head(salida[:, -1])


def entrenar(clase, semilla, epocas=400, paciencia=40):
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    red = clase()
    opt = torch.optim.Adam(red.parameters(), lr=1e-3, weight_decay=1e-4)
    Xf_t, Yf_t = torch.tensor(Xf), torch.tensor(Yf)
    Xv_t, Yv_t = torch.tensor(Xv), torch.tensor(Yv)
    mejor, estado, sin_mejora, curva = np.inf, None, 0, []
    for ep in range(epocas):
        red.train()
        perm = torch.randperm(len(Xf_t))
        for b in range(0, len(perm), 64):
            j = perm[b:b + 64]
            opt.zero_grad()
            perdida = torch.mean((red(Xf_t[j]) - Yf_t[j]) ** 2)
            perdida.backward()
            opt.step()
        red.eval()
        with torch.no_grad():
            pv = torch.mean((red(Xv_t) - Yv_t) ** 2).item()
        curva.append(pv)
        if pv < mejor - 1e-4:
            mejor, estado, sin_mejora = pv, {k: v.clone() for k, v in red.state_dict().items()}, 0
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    red.load_state_dict(estado)
    return red, curva


REDES = {}


def predecir_red(clase, nombre, semillas=range(5)):
    t0 = time.time()
    preds, curvas, redes = [], [], []
    for s in semillas:
        red, curva = entrenar(clase, s)
        red.eval()
        curvas.append(curva)
        redes.append(red)
        with torch.no_grad():
            preds.append(red(torch.tensor(Xt)).numpy())
    REDES[nombre] = redes
    print(f"{nombre}: {time.time() - t0:.0f} s · épocas hasta early stopping: {[len(c) for c in curvas]}")
    return pd.DataFrame(np.mean(preds, axis=0) * sd + mu, index=o_test, columns=range(1, H + 1)), curvas


pron["CNN 1D"], curvas_cnn = predecir_red(CNN1D, "CNN 1D")
pron["LSTM"], curvas_lstm = predecir_red(LSTMNet, "LSTM")

t0 = time.time()
mlps = [MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3, max_iter=2000,
                     early_stopping=True, validation_fraction=0.2, random_state=s).fit(
    np.concatenate([Xf, Xv]).reshape(len(Xf) + len(Xv), -1), np.concatenate([Yf, Yv])) for s in range(5)]
pron["MLP"] = pd.DataFrame(np.mean([m_.predict(Xt.reshape(len(Xt), -1)) for m_ in mlps], axis=0) * sd + mu,
                           index=o_test, columns=range(1, H + 1))
print(f"MLP: {time.time() - t0:.0f} s")

# %% [markdown]
# ### Prophet (Clase 12)
#
# Prophet descompone la serie en tendencia + estacionalidad + eventos. No tiene componente autorregresiva, así que
# **no "recuerda" el estado reciente** más allá de la tendencia. Lo esperable es que se parezca a la climatología.
# Por eso lo incluimos: muestra para qué *no* sirve un modelo. Lo reajustamos en cada origen.

# %%
from prophet import Prophet

t0 = time.time()
filas = {}
for o in ORIGENES:
    df = pd.DataFrame({"ds": x.loc[:o].index, "y": x.loc[:o].to_numpy()})
    mp = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                 changepoint_prior_scale=0.5, uncertainty_samples=0).fit(df)
    fut = pd.DataFrame({"ds": pd.date_range(o + pd.DateOffset(months=1), periods=H, freq="MS")})
    filas[o] = mp.predict(fut)["yhat"].to_numpy()
pron["Prophet"] = pd.DataFrame.from_dict(filas, orient="index", columns=range(1, H + 1))
print(f"Prophet: {time.time() - t0:.0f} s ({len(ORIGENES)} ajustes)")

# Ensamble simple: promedio de ARIMA (mejor a corto plazo) y MLP (mejor a largo plazo)
pron["ARIMA + MLP"] = (pron["ARIMA"].reindex(ORIGENES) + pron["MLP"].reindex(ORIGENES)) / 2

# %% [markdown]
# ## 4. Resultados por horizonte

# %%
orden_m = ["Persistencia", "Climatología", "ARIMA", "MLP", "CNN 1D", "LSTM", "Prophet", "ARIMA + MLP"]
tab = []
for nombre in orden_m:
    for h in range(1, H + 1):
        mt = metricas(Yobs[h], pron[nombre].reindex(ORIGENES)[h])
        tab.append({"modelo": nombre, "h": h, **mt})
tab = pd.DataFrame(tab)
rmse = tab.pivot(index="modelo", columns="h", values="RMSE").loc[orden_m]
acc = tab.pivot(index="modelo", columns="h", values="r").loc[orden_m]
print("RMSE (°C) por horizonte (meses):")
print(rmse.round(3).to_string())
print("\nACC (correlación pronóstico-observado):")
print(acc.round(3).to_string())

fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
for nombre in orden_m:
    estilo_l = dict(color=MODELO_COLOR[nombre], lw=2.4 if nombre in ("ARIMA", "ARIMA + MLP") else 1.4,
                    ls="--" if nombre in ("Persistencia", "Climatología") else "-")
    axs[0].plot(rmse.columns, rmse.loc[nombre], marker="o", ms=3.5, label=nombre, **estilo_l)
    if nombre != "Climatología":
        axs[1].plot(acc.columns, acc.loc[nombre], marker="o", ms=3.5, **estilo_l)
axs[0].set_title("RMSE (°C), menor es mejor")
axs[1].set_title("Correlación de anomalías (ACC), mayor es mejor")
for ax in axs:
    ax.set_xlabel("horizonte (meses)")
    ax.set_xticks(range(1, H + 1))
axs[1].axhline(0.6, color=COLOR["tinta2"], lw=0.8, ls=":")
axs[1].text(1.1, 0.61, "ACC 0,6: umbral usual de pronóstico útil", fontsize=8, color=COLOR["tinta2"])
axs[0].legend(fontsize=8, ncol=2)
fig.suptitle("Pronóstico del Niño 3.4, test 2000–2026 (origen móvil)", x=0.01, ha="left", fontweight="bold")
fig.savefig(FIG / "12_skill_modelos_enso.png")
plt.show()

# %%
dm = []
for nombre in ["Persistencia", "MLP", "CNN 1D", "LSTM", "Prophet", "ARIMA + MLP"]:
    for h in (1, 3, 6, 9):
        e_m = (Yobs[h] - pron[nombre].reindex(ORIGENES)[h]).to_numpy()
        e_a = (Yobs[h] - pron["ARIMA"].reindex(ORIGENES)[h]).to_numpy()
        r = dm_test(e_m, e_a, h=h)
        dm.append({"modelo vs ARIMA": nombre, "h": h, "ΔRMSE": rmse.loc[nombre, h] - rmse.loc["ARIMA", h],
                   "DM": r["DM"], "p": r["p"]})
dm = pd.DataFrame(dm)
print(dm.round(4).to_string(index=False))

# %% [markdown]
# ### La "barrera de predictibilidad de otoño" (boreal spring barrier)
#
# El ENSO es más difícil de pronosticar cuando el pronóstico **cruza el otoño austral** (mar–may). En esa época las
# anomalías son chicas y el sistema "decide" su próxima fase. Un pronóstico hecho en el verano (dic–feb) para el invierno
# siguiente casi no tiene skill. Uno hecho en invierno para el verano siguiente, en cambio, sí lo tiene, y esa es justo
# la ventana de la decisión de siembra.

# %%
mejor_red = min(["MLP", "CNN 1D", "LSTM"], key=lambda n: rmse.loc[n, 6])
mapa = {}
for nombre in ["ARIMA", mejor_red]:
    M = pd.DataFrame(index=range(1, 13), columns=range(1, H + 1), dtype=float)
    P = pron[nombre].reindex(ORIGENES)
    for m_ in range(1, 13):
        sel = ORIGENES.month == m_
        for h in range(1, H + 1):
            M.loc[m_, h] = metricas(Yobs[h][sel], P[h][sel])["r"]
    mapa[nombre] = M
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
for ax, (nombre, M) in zip(axs, mapa.items()):
    im = ax.imshow(M.to_numpy(float), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(H), range(1, H + 1))
    ax.set_yticks(range(12), MESES)
    ax.set_xlabel("horizonte (meses)")
    ax.set_title(f"ACC por mes de origen: {nombre}")
    ax.grid(False)
    for i in range(12):
        for j in range(H):
            v = M.iloc[i, j]
            ax.text(j, i, f"{v:.1f}".replace(".", ","), ha="center", va="center", fontsize=7,
                    color=COLOR["fondo"] if v > 0.6 else COLOR["tinta"])
axs[0].set_ylabel("mes de origen del pronóstico")
fig.colorbar(im, ax=axs, pad=0.01, label="ACC")
fig.savefig(FIG / "13_barrera_predictibilidad.png")
plt.show()
for n_, M in mapa.items():
    print(f"ACC a 6 meses, {n_}: desde enero = {M.loc[1, 6]:.2f} · desde julio = {M.loc[7, 6]:.2f} · desde septiembre = {M.loc[9, 6]:.2f}")

# %% [markdown]
# ## 5. Pronóstico real fuera de muestra: El Niño 2026/27
#
# El último dato mensual del Niño 3.4 es de junio de 2026. Desde ahí, cada modelo pronostica jul 2026 a mar 2027.
# Hay una **verificación real** disponible: el ONI de jul–sep 2026 (+2,16 °C), que se publicó después del último dato mensual.

# %%
o_fin = x.index[-1]
fut = pd.date_range(o_fin + pd.DateOffset(months=1), periods=H, freq="MS")
res_full = ARIMA(x, order=ORDEN, trend="c").fit()
fc = res_full.get_forecast(H)
final = pd.DataFrame({"ARIMA": fc.predicted_mean.to_numpy()}, index=fut)
ic80 = fc.conf_int(alpha=0.2).to_numpy()
ic95 = fc.conf_int(alpha=0.05).to_numpy()
Xlast, _, _ = ventanas([o_fin])
with torch.no_grad():
    for nombre in ["CNN 1D", "LSTM"]:
        final[nombre] = np.mean([red(torch.tensor(Xlast)).numpy()[0] for red in REDES[nombre]], axis=0) * sd + mu
final["MLP"] = np.mean([m_.predict(Xlast.reshape(1, -1))[0] for m_ in mlps], axis=0) * sd + mu
final["ARIMA + MLP"] = (final["ARIMA"] + final["MLP"]) / 2
oni_obs = enso["oni"].loc["2026-07":]

fig, ax = plt.subplots(figsize=(11, 4))
hist = x.loc["2024-01":]
ax.plot(hist.index, hist, color=COLOR["tinta"], lw=1.8, label="Niño 3.4 observado (mensual)")
ax.plot(oni_obs.index, oni_obs, color=COLOR["tinta"], lw=0, marker="s", ms=6, label="ONI publicado (trimestre móvil)")
ax.fill_between(fut, ic95[:, 0], ic95[:, 1], color=COLOR["tinta"], alpha=0.08, lw=0, label="ARIMA IC 95 %")
ax.fill_between(fut, ic80[:, 0], ic80[:, 1], color=COLOR["tinta"], alpha=0.15, lw=0, label="ARIMA IC 80 %")
for nombre in ["ARIMA", "MLP", "CNN 1D", "LSTM", "ARIMA + MLP"]:
    ax.plot(fut, final[nombre], color=MODELO_COLOR[nombre], lw=2, marker="o", ms=3, label=nombre)
ax.axhline(0.5, color=COLOR["nino"], lw=0.8, ls=":")
ax.axhline(0, color=COLOR["tinta2"], lw=0.6)
ax.set_title(f"Pronóstico del Niño 3.4 desde {o_fin:%b %Y} y verificación con el ONI publicado")
ax.set_ylabel("anomalía (°C)")
ax.legend(fontsize=8, ncol=2, loc="upper left")
fig.savefig(FIG / "14_pronostico_2026.png")
plt.show()

print(final.round(2).T.to_string())
jas = final.loc["2026-07":"2026-09"].mean()
djf = final.loc["2026-12":"2027-02"].mean()
print("\npronóstico JAS 2026 (observado +2,16):", jas.round(2).to_dict())
print("pronóstico DJF 2026/27:", djf.round(2).to_dict())

# %%
salida = {"orden_arima": ORDEN, "rmse": rmse.round(4).to_dict(), "acc": acc.round(4).to_dict(),
          "dm": dm.round(5).to_dict("records"), "mejor_red": mejor_red,
          "barrera": {n: {"ene_h6": float(M.loc[1, 6]), "jul_h6": float(M.loc[7, 6]), "sep_h6": float(M.loc[9, 6])} for n, M in mapa.items()},
          "jas_2026_pronostico": jas.round(3).to_dict(), "djf_2026_27_pronostico": djf.round(3).to_dict(),
          "jas_2026_observado": float(enso["oni"].loc["2026-08-01"])}
(RES / "etapa3a_enso.json").write_text(json.dumps(salida, indent=2, default=float, ensure_ascii=False))
final.to_csv(RES / "pronostico_nino34_2026.csv")

# %% [markdown]
# ## 6. Conclusiones (Clases 11 y 12)
#
# - **STL:** la TSM del Niño 3.4 tiene una tendencia de unos +0,4 °C (1950–2026) y un ciclo anual marcado. El resto
#   de la STL reproduce la anomalía oficial (r = 0,95).
# - **ARIMA(2,0,2)**, elegido por AICc y consistente con la ACF y la PACF: el par AR complejo (φ₁ = 1,86, φ₂ = −0,88) es
#   un oscilador amortiguado, es decir, la cuasi-periodicidad del ENSO que se vio en el espectro de la Etapa 1. Los residuos son
#   ruido blanco (Ljung-Box p = 0,35 y 0,74; DW = 2,0), con colas pesadas.
# - **Horizontes de 1 y 2 meses:** gana el ARIMA, reajustado cada año. Las redes, entrenadas una sola vez hasta 1989,
#   pierden contra la persistencia a 1 mes.
# - **Horizontes de 4 a 9 meses:** el MLP supera al ARIMA. A 6 meses su ACC es 0,61 contra 0,44 (DM p = 0,08).
#   La CNN 1D y la LSTM quedan entre las dos, sin diferencia significativa con el ARIMA.
#   **Con 448 muestras de entrenamiento, una arquitectura más compleja no mejora.**
# - **Ensamble ARIMA + MLP:** es el mejor a partir de 3 meses y le gana al ARIMA con DM p < 0,01 en h = 3, 6 y 9.
# - **Prophet** fracasa (ACC < 0): no tiene memoria autorregresiva y el ENSO no tiene ciclo anual en la media.
#   Es un buen ejemplo de un modelo que no corresponde al problema.
# - **Barrera de predictibilidad:** a 6 meses, la ACC del ARIMA es de unos 0,0 si el pronóstico arranca en enero y
#   cruza el otoño, contra unos 0,7 si arranca en julio. El MLP atenúa la barrera (0,3 desde enero).
#   Por suerte para el agro, la decisión de siembra (sep–oct) cae del lado bueno.
# - **El caso 2026:** desde junio, todos los modelos **subestimaron** el JAS observado (+2,16). El más cercano fue el MLP, con +1,71.
#   Para dic–feb 2026/27, el ensamble pronostica unos +1,5 °C, es decir, un Niño fuerte que persiste en el período crítico de la soja.
