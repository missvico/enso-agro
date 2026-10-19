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
# # Etapa 1 — Análisis exploratorio: ENSO, lluvia y rendimientos en la Región Pampeana
#
# **TP Data Mining de Series Temporales — MEDyDC, FCEyN-UBA**
#
# Objetivo de esta etapa: caracterizar las series (dominio del tiempo, de la frecuencia y
# tiempo-frecuencia), dejar los rendimientos en una forma estacionaria y obtener una
# **primera evidencia** de la relación ENSO → lluvia → rendimiento, antes de modelar.
#
# | Sección | Métodos | Clase |
# |---|---|---|
# | 1. ONI | representación, MA3 | 2, 4 |
# | 2. Espectro del Niño 3.4 | FFT, k·Fs/N, espectro AR, Parseval | 3, 11 |
# | 3. Wavelets | DWT (Mallat) y CWT | 10 |
# | 4. Rendimientos | tendencia (lineal, MA, LOWESS), ADF+KPSS, ACF | 4, 11 |
# | 5. ENSO ↔ rendimiento | perfil de correlación por trimestre, fases | 6 |
# | 6. Mecanismo mensual | CCF y Granger Niño 3.4 → lluvia pampeana | 6, 8 |

# %%
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from scipy import stats
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.stattools import acf, grangercausalitytests

RAIZ = Path.cwd().resolve()
if RAIZ.name == "notebooks":
    RAIZ = RAIZ.parent
sys.path.insert(0, str(RAIZ / "src"))
from utils_ts import (COLOR, CULTIVO_COLOR, CULTIVO_NOMBRE, adf_kpss, anomalia_relativa, ccf,
                      estilo, tendencia)

warnings.filterwarnings("ignore", category=FutureWarning)
estilo()
PROC, FIG = RAIZ / "data" / "processed", RAIZ / os.environ.get("FIG_DIR", "figures")
FIG.mkdir(exist_ok=True, parents=True)

enso = pd.read_parquet(PROC / "enso_mensual.parquet")
clima = pd.read_parquet(PROC / "clima_pampeano_mensual.parquet")
rend = pd.read_parquet(PROC / "rendimientos_anuales.parquet")
camp = pd.read_parquet(PROC / "campanias.parquet")
pd.set_option("display.width", 140)

# %% [markdown]
# ## 1. El índice ONI
#
# El ONI es la anomalía de temperatura superficial del mar en la región Niño 3.4, **promediada
# en trimestres móviles**: es literalmente un filtro MA3 (Clase 4) aplicado a la serie mensual.
# Fases (NOAA): Niño si ONI ≥ +0,5 °C y Niña si ONI ≤ −0,5 °C durante ≥ 5 trimestres seguidos.

# %%
oni = enso["oni"].dropna()
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.fill_between(oni.index, 0, oni, where=oni >= 0, color=COLOR["nino"], lw=0, interpolate=True, label="Fase cálida (Niño)")
ax.fill_between(oni.index, 0, oni, where=oni < 0, color=COLOR["nina"], lw=0, interpolate=True, label="Fase fría (Niña)")
for u in (0.5, -0.5):
    ax.axhline(u, color=COLOR["tinta2"], lw=0.6)
ax.axhline(0, color=COLOR["tinta"], lw=0.8)
for fecha, txt in [("1983-01", "1982/83"), ("1998-01", "1997/98"), ("2015-12", "2015/16")]:
    ax.annotate(txt, (pd.Timestamp(fecha), oni[fecha].iloc[0]), xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=8, color=COLOR["tinta2"])
ax.annotate(f"{oni.index[-1]:%b %Y}: {oni.iloc[-1]:+.2f}", (oni.index[-1], oni.iloc[-1]), xytext=(-6, 4),
            textcoords="offset points", ha="right", fontsize=8, color=COLOR["tinta"])
ax.set_title("Oceanic Niño Index (ONI), 1950–2026")
ax.set_ylabel("anomalía TSM (°C)")
ax.legend(loc="lower left", ncol=2, fontsize=8)
fig.savefig(FIG / "01_oni.png")
plt.show()

print(enso.loc["2025-09":, ["oni", "n34_anom", "fase"]].T)

# %%
# ¿El ONI es un MA3 centrado del Niño 3.4 mensual? (Clase 4)
ma3 = enso["n34_anom"].rolling(3, center=True).mean()
par = pd.concat([enso["oni"], ma3], axis=1, keys=["oni", "ma3"]).dropna()
print(f"corr(ONI, MA3(Niño3.4)) = {par.corr().iloc[0, 1]:.4f}   "
      f"RMSE = {np.sqrt(((par.oni - par.ma3) ** 2).mean()):.3f} °C")
print("(la diferencia residual viene de que el ONI usa climatologías base de 30 años que se actualizan cada 5)")

# %% [markdown]
# ## 2. Dominio de la frecuencia: ¿cada cuánto ocurre El Niño?
#
# Usamos la anomalía **mensual** de Niño 3.4 (sin el MA3, que atenuaría altas frecuencias).
# El eje se calibra con k·Fs/N (Clase 3) con Fs = 12 muestras/año. Comparamos el periodograma
# crudo con el espectro de un modelo AR (Clase 11, `spec.ar`) y con un **ruido rojo AR(1)** al 95 %
# como referencia de significancia (Torrence & Compo, 1998).

# %%
x = enso["n34_anom"].dropna()
x = x - x.mean()
N, Fs = len(x), 12.0
X = np.fft.fft(x.to_numpy())
k = np.arange(1, N // 2)
f = k * Fs / N                      # ciclos por año
P = (np.abs(X[k]) ** 2) / N         # periodograma (escala de Parseval)
print(f"N = {N} meses ({x.index[0]:%Y-%m} a {x.index[-1]:%Y-%m}), ΔF = {Fs / N:.4f} ciclos/año")
print(f"Parseval: Σx² = {np.sum(x ** 2):.1f}   (1/N)Σ|X|² = {np.sum(np.abs(X) ** 2) / N:.1f}")

# espectro AR de orden elegido por AIC
orden = ar_select_order(x.to_numpy(), maxlag=36, ic="aic").ar_lags
p = max(orden) if orden else 1
ar = AutoReg(x.to_numpy(), lags=p).fit()
phi, s2 = ar.params[1:], ar.sigma2
w = 2 * np.pi * f / Fs
H = 1 - sum(phi[j] * np.exp(-1j * w * (j + 1)) for j in range(p))
S_ar = s2 / np.abs(H) ** 2

# ruido rojo AR(1) y su banda 95 % (chi² con 2 g.l.)
r1 = x.autocorr(1)
S_rojo = np.var(x) * (1 - r1 ** 2) / (1 + r1 ** 2 - 2 * r1 * np.cos(w))
# periodograma suavizado con una ventana de Hanning de 7 armónicas (Clase 9)
h = np.hanning(9)[1:-1]
h = h / h.sum()
P_suave = np.convolve(P, h, mode="same")
# banda 95 %: cada ordenada cruda es chi²(2)/2; al suavizar, g.l. = 2 / Σh²
gl = 2 / np.sum(h ** 2)
S_rojo95 = S_rojo * stats.chi2.ppf(0.95, gl) / gl
print(f"r1 = {r1:.3f}; grados de libertad del periodograma suavizado = {gl:.1f}")

periodo = 1 / f
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(periodo, P, color=COLOR["neutral"], lw=0.8, label="periodograma (FFT)")
ax.plot(periodo, P_suave, color=COLOR["serie"], lw=1.8, label="periodograma suavizado (Hanning)")
ax.plot(periodo, S_ar, color=COLOR["tendencia"], lw=1.8, label=f"espectro AR({p})")
ax.plot(periodo, S_rojo95, color=COLOR["tinta2"], lw=1, ls="--", label="ruido rojo AR(1), 95 %")
ax.axvspan(2, 7, color=COLOR["nino"], alpha=0.07, lw=0)
ax.set_yscale("log")
ax.set_ylim(1e-2, 60)
ax.text(3.7, 40, "banda ENSO 2–7 años", ha="center", fontsize=8, color=COLOR["tinta2"])
ax.set_xscale("log")
ax.set_xlim(0.4, 40)
ax.set_xticks([0.5, 1, 2, 3, 5, 7, 10, 20, 40], ["0,5", "1", "2", "3", "5", "7", "10", "20", "40"])
ax.set_xlabel("período (años)")
ax.set_ylabel("potencia (escala log)")
ax.set_title("Espectro de la anomalía Niño 3.4 mensual, 1950–2026")
ax.legend(fontsize=8, loc="lower center")
fig.savefig(FIG / "02_espectro_nino34.png")
plt.show()

top = pd.DataFrame({"periodo_anios": periodo, "P_suave": P_suave, "sig95": P_suave > S_rojo95})
print(top.sort_values("P_suave", ascending=False).head(5).round(2).to_string(index=False))
banda = (periodo >= 2) & (periodo <= 7)
print(f"fracción de la varianza en 2–7 años: {P[banda].sum() / P.sum():.1%}")

# %% [markdown]
# ## 3. Tiempo-frecuencia: ENSO no es estacionario en frecuencia (Clase 10)
#
# El espectro de Fourier promedia toda la serie y no dice **cuándo** domina cada período.
# (a) **DWT** con el algoritmo piramidal de Mallat (Daubechies 4, como `wd(..., filter.number=4)` en
# la clase): con datos mensuales, el detalle D*j* captura períodos de 2^j a 2^(j+1) meses.
# (b) **CWT Morlet**: escalograma continuo, la forma estándar de mostrar ENSO.

# %%
niveles = 7
coefs = pywt.wavedec(x.to_numpy(), "db4", level=niveles)
componentes = {}
for i in range(len(coefs)):
    c = [np.zeros_like(cc) for cc in coefs]
    c[i] = coefs[i]
    nombre = f"A{niveles}" if i == 0 else f"D{niveles - i + 1}"
    componentes[nombre] = pywt.waverec(c, "db4")[:N]
energia = {n: np.sum(v ** 2) for n, v in componentes.items()}
tot = sum(energia.values())
tabla_dwt = pd.DataFrame({
    "nivel": list(energia),
    "periodos (años)": [f"> {2 ** niveles / 12:.1f}" if n.startswith("A") else
                        f"{2 ** int(n[1:]) / 12:.1f}–{2 ** (int(n[1:]) + 1) / 12:.1f}" for n in energia],
    "% energía": [100 * e / tot for e in energia.values()],
}).round(1)
print(tabla_dwt.to_string(index=False))

fig, axs = plt.subplots(4, 1, figsize=(11, 6.5), sharex=True)
t = x.index
axs[0].plot(t, x, color=COLOR["tinta2"], lw=0.8)
axs[0].set_title("Descomposición DWT (db4) de Niño 3.4 — algoritmo de Mallat")
axs[0].set_ylabel("serie")
for ax, n in zip(axs[1:], ["D4", "D5", "D6"]):
    ax.plot(t, componentes[n], color=COLOR["serie"], lw=1.2)
    fila = tabla_dwt[tabla_dwt.nivel == n].iloc[0]
    ax.set_ylabel(n)
    ax.text(0.005, 0.85, f"{fila['periodos (años)']} años · {fila['% energía']}% energía", transform=ax.transAxes,
            fontsize=8, color=COLOR["tinta2"])
fig.savefig(FIG / "03a_dwt_nino34.png")
plt.show()

# %%
wavelet, B = "cmor1.5-1.0", 1.5
periodos_obj = np.geomspace(0.5, 16, 80)                     # años
escalas = pywt.frequency2scale(wavelet, 1 / (periodos_obj * 12))  # frecuencia en ciclos/muestra
cw, freqs = pywt.cwt(x.to_numpy(), escalas, wavelet, sampling_period=1 / 12)
potencia = np.abs(cw) ** 2 / np.var(x)
per_cwt = 1 / freqs
anios = x.index.year + (x.index.month - 0.5) / 12
# cono de influencia: e-folding de la envolvente gaussiana = escala·sqrt(B/2) muestras
coi_meses = escalas * np.sqrt(B / 2)
dist_borde = np.minimum(np.arange(N), np.arange(N)[::-1]) / 12

fig, ax = plt.subplots(figsize=(11, 4.4))
niveles_pot = np.linspace(0, np.quantile(potencia, 0.995), 21)
im = ax.contourf(anios, per_cwt, potencia, levels=niveles_pot, cmap="Blues", extend="max")
coi = np.interp(dist_borde, coi_meses / 12, per_cwt)
ax.fill_between(anios, coi, per_cwt.max(), color=COLOR["fondo"], alpha=0.65, lw=0)
ax.set_yscale("log")
ax.set_yticks([0.5, 1, 2, 4, 8, 16], ["0,5", "1", "2", "4", "8", "16"])
ax.set_ylim(per_cwt.max(), per_cwt.min())
ax.set_ylabel("período (años)")
ax.set_title("Escalograma CWT (Morlet) de Niño 3.4 — zona velada: cono de influencia")
ax.grid(False)
cb = fig.colorbar(im, ax=ax, pad=0.01)
cb.set_label("potencia / varianza")
cb.set_ticks(range(0, int(niveles_pot[-1]) + 1, 2))
fig.savefig(FIG / "03b_cwt_nino34.png")
plt.show()

banda_cwt = (per_cwt >= 2) & (per_cwt <= 7)
pot_banda = pd.Series(potencia[banda_cwt].mean(axis=0), index=x.index)
print("potencia media en la banda 2–7 años por década:")
print(pot_banda.groupby((pot_banda.index.year // 10) * 10).mean().round(2).to_string())

# %% [markdown]
# ## 4. Rendimientos: tendencia tecnológica y estacionariedad
#
# Trabajamos con la **Región Pampeana** (Buenos Aires, Córdoba, Santa Fe, Entre Ríos y La Pampa;
# producción / superficie cosechada) desde la campaña **1970/71**, cuando la soja ya tiene
# superficie relevante. La tendencia se estima con tres filtros pasa-bajos (lineal, MA9 centrada,
# LOWESS) y se trabaja con la **anomalía relativa**: 100·(y / tendencia − 1).

# %%
INICIO, FIN = 1970, 2024
reg = rend[(rend.region == "pampeana") & rend.campania.between(INICIO, FIN)]
Y = reg.pivot(index="campania", columns="cultivo", values="rend_kgha")[["soja", "maiz", "trigo"]]

anom, tends = {}, {}
for c in Y:
    tends[c] = {m: tendencia(Y[c], m) for m in ("lineal", "ma", "lowess")}
    anom[c] = {m: anomalia_relativa(Y[c], tends[c][m]) for m in tends[c]}
A = pd.DataFrame({c: anom[c]["lowess"] for c in Y})        # anomalía principal (LOWESS)

fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True)
for ax, c in zip(axs, Y):
    ax.plot(Y.index, Y[c] / 1000, color=COLOR["tinta2"], lw=1.2, marker="o", ms=2.5, label="observado")
    ax.plot(Y.index, tends[c]["lowess"] / 1000, color=COLOR["tendencia"], lw=2, label="tendencia LOWESS")
    ax.plot(Y.index, tends[c]["lineal"] / 1000, color=COLOR["tendencia"], lw=0.8, ls=":", label="tendencia lineal")
    ax.set_title(CULTIVO_NOMBRE[c])
    ax.set_ylabel("t/ha")
axs[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Rendimiento en la Región Pampeana, campañas 1970/71–2024/25", x=0.01, ha="left", fontweight="semibold")
fig.savefig(FIG / "04a_rendimientos.png")
plt.show()

# robustez: las anomalías casi no dependen del filtro elegido
print("corr. entre anomalías según filtro de tendencia:")
for c in Y:
    d = pd.DataFrame(anom[c]).dropna()
    print(f"  {c:6s} lowess~ma: {d.lowess.corr(d.ma):.2f}   lowess~lineal: {d.lowess.corr(d.lineal):.2f}")

# %%
filas = []
for c in Y:
    for nombre, serie, reg_ in [("nivel (kg/ha)", Y[c], "c"), ("nivel, test con tendencia", Y[c], "ct"),
                                ("Δ log (tasa anual)", np.log(Y[c]).diff(), "c"),
                                ("anomalía LOWESS (%)", A[c], "c")]:
        filas.append({"cultivo": c, "serie": nombre, **adf_kpss(serie, reg_)})
est = pd.DataFrame(filas)
print(est.to_string(index=False))

# %%
fig, axs = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
for ax, c in zip(axs, Y):
    a = A[c].dropna()
    r = acf(a, nlags=10)
    ax.vlines(range(len(r)), 0, r, color=CULTIVO_COLOR[c], lw=2)
    ax.axhline(0, color=COLOR["tinta"], lw=0.8)
    for s in (1, -1):
        ax.axhline(s * 1.96 / np.sqrt(len(a)), color=COLOR["tinta2"], lw=0.8, ls="--")
    ax.set_title(f"ACF anomalía — {CULTIVO_NOMBRE[c]}")
    ax.set_xlabel("lag (campañas)")
fig.savefig(FIG / "04b_acf_anomalias.png")
plt.show()

# %% [markdown]
# ## 5. ENSO ↔ rendimiento: ¿en qué trimestre importa el ONI?
#
# A escala anual, una CCF con lags en **años** no tiene sentido físico (el efecto ocurre dentro de
# la misma campaña). Lo que sí tiene sentido es barrer el **trimestre** en que se mide el ONI,
# desde el invierno previo a la siembra hasta el otoño de cosecha: es una CCF a escala mensual
# sobre una respuesta anual. Línea punteada: umbral ±1,96/√n (como `ccf()` de R).

# %%
A_long = A.stack().rename("anom").reset_index().rename(columns={"level_1": "cultivo"})
cols_oni = [c for c in camp.columns if c.startswith("oni_")]
oni_camp = camp.drop_duplicates("campania").set_index("campania")[cols_oni]
etiq = [c.split("_")[1] + ("" if c.endswith("_0") else "⁺") for c in cols_oni]

perfil = {}
for c in Y:
    d = A[c].dropna().to_frame("anom").join(oni_camp)
    perfil[c] = pd.Series({col: d["anom"].corr(d[col], method="spearman") for col in cols_oni})
perfil = pd.DataFrame(perfil)
n_camp = A.dropna().shape[0]
umbral = 1.96 / np.sqrt(n_camp)

fig, ax = plt.subplots(figsize=(11, 4.2))
for c in Y:
    ax.plot(range(len(cols_oni)), perfil[c], color=CULTIVO_COLOR[c], marker="o", ms=4, lw=1.8, label=CULTIVO_NOMBRE[c])
    j = int(np.argmax(np.abs(perfil[c].to_numpy())))
    ax.annotate(f"{perfil[c].iloc[j]:+.2f}", (j, perfil[c].iloc[j]), xytext=(0, 7), textcoords="offset points",
                ha="center", fontsize=8, color=COLOR["tinta"])
for s in (1, -1):
    ax.axhline(s * umbral, color=COLOR["tinta2"], lw=0.8, ls="--")
ax.axhline(0, color=COLOR["tinta"], lw=0.8)
ax.set_xticks(range(len(cols_oni)), etiq)
ax.axvspan(7.5, 12.5, color=COLOR["neutral"], alpha=0.12, lw=0)
ax.text(10, ax.get_ylim()[0] + 0.02, "año de cosecha (⁺)", ha="center", fontsize=8, color=COLOR["tinta2"])
ax.set_ylabel("ρ de Spearman")
ax.set_xlabel("trimestre en que se mide el ONI")
ax.set_title(f"Correlación anomalía de rendimiento vs ONI por trimestre (pampeana, n = {n_camp})")
ax.legend(fontsize=9, loc="upper right")
fig.savefig(FIG / "05a_perfil_correlacion.png")
plt.show()
print(perfil.round(2).set_index(pd.Index(etiq)).T.to_string())

# %% [markdown]
# **Ventanas fijadas a priori por fenología** (para no elegir el máximo a posteriori):
# - Soja y maíz (cultivos de verano): ONI medio **OND–NDJ–DJF** (siembra → floración).
# - Trigo (cultivo de invierno): ONI medio **JAS–ASO–SON** (macollaje → llenado).

# %%
VENTANA = {"soja": ["oni_OND_0", "oni_NDJ_0", "oni_DJF_1"],
           "maiz": ["oni_OND_0", "oni_NDJ_0", "oni_DJF_1"],
           "trigo": ["oni_JAS_0", "oni_ASO_0", "oni_SON_0"]}


def fase_campania(v: float) -> str:
    return "Niño" if v >= 0.5 else ("Niña" if v <= -0.5 else "Neutral")


def fmt_p(p: float) -> str:
    return "p<0,001" if p < 0.001 else f"p={p:.3f}".replace(".", ",")


tabla = []
for c in Y:
    d = A[c].dropna().to_frame("anom").join(oni_camp[VENTANA[c]].mean(axis=1).rename("oni_ventana"))
    d["fase"] = d["oni_ventana"].map(fase_campania)
    d["cultivo"] = c
    tabla.append(d.reset_index())
fases = pd.concat(tabla, ignore_index=True)

orden_f = ["Niña", "Neutral", "Niño"]
col_f = {"Niña": COLOR["nina"], "Neutral": COLOR["neutral"], "Niño": COLOR["nino"]}
fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
rng = np.random.default_rng(0)
resumen = []
for ax, c in zip(axs, Y):
    d = fases[fases.cultivo == c]
    for i, fz in enumerate(orden_f):
        v = d.loc[d.fase == fz, "anom"]
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=22, color=col_f[fz], edgecolor=COLOR["fondo"], lw=0.8, zorder=3)
        ax.hlines(v.median(), i - 0.25, i + 0.25, color=COLOR["tinta"], lw=2, zorder=4)
        resumen.append({"cultivo": c, "fase": fz, "n": len(v), "mediana_%": v.median(), "media_%": v.mean(),
                        "p(anom<-10%)": (v < -10).mean()})
    kw = stats.kruskal(*[d.loc[d.fase == fz, "anom"] for fz in orden_f])
    mw = stats.mannwhitneyu(d.loc[d.fase == "Niña", "anom"], d.loc[d.fase != "Niña", "anom"], alternative="less")
    resumen[-1]["KW_p"], resumen[-1]["MW_Niña<resto_p"] = kw.pvalue, mw.pvalue
    ax.axhline(0, color=COLOR["tinta"], lw=0.8)
    ax.set_xticks(range(3), orden_f)
    ax.set_title(f"{CULTIVO_NOMBRE[c]}\nKruskal-Wallis {fmt_p(kw.pvalue)} · Niña < resto {fmt_p(mw.pvalue)}", fontsize=10)
axs[0].set_ylabel("anomalía de rendimiento (%)")
fig.suptitle("Anomalía de rendimiento pampeano según fase ENSO de la ventana fenológica (línea: mediana)",
             x=0.01, ha="left", fontweight="semibold")
fig.savefig(FIG / "05b_fases.png")
plt.show()
print(pd.DataFrame(resumen).round(3).to_string(index=False))

# %%
print("Peores 6 campañas por cultivo (anomalía %, ONI de la ventana):")
for c in Y:
    d = fases[fases.cultivo == c].nsmallest(6, "anom")
    print(f"  {c:6s}", ", ".join(f"{int(r.campania)}/{str(int(r.campania) + 1)[-2:]} ({r.anom:+.0f}%, {r.oni_ventana:+.1f})"
                                  for r in d.itertuples()))

# %% [markdown]
# ## 6. El mecanismo a escala mensual: Niño 3.4 → lluvia pampeana (1981–2025)
#
# Acá sí tenemos dos series mensuales largas (540 meses), que es el escenario para el que están
# pensados la CCF (Clase 6) y el test de Granger (Clase 8). Lluvia: promedio de 11 localidades
# pampeanas (NASA POWER), como anomalía estandarizada por mes calendario.

# %%
m = pd.concat([enso["n34_anom"], clima["precip_z"]], axis=1).dropna()
print("Estacionariedad mensual:")
print(pd.DataFrame({s: adf_kpss(m[s]) for s in m}).T.to_string())

cc = ccf(m["n34_anom"], m["precip_z"], max_lag=12)
primavera = m.index.month.isin([10, 11, 12])
r_prim = {k: m["n34_anom"].shift(k)[primavera].corr(m["precip_z"][primavera]) for k in range(0, 7)}

fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [2, 1]})
axs[0].vlines(cc.lag, 0, cc.r, color=COLOR["serie"], lw=2.2)
axs[0].plot(cc.lag, cc.umbral95, color=COLOR["tinta2"], lw=0.8, ls="--")
axs[0].plot(cc.lag, -cc.umbral95, color=COLOR["tinta2"], lw=0.8, ls="--")
axs[0].axhline(0, color=COLOR["tinta"], lw=0.8)
axs[0].set_xlabel("lag k (meses)  ·  k > 0: Niño 3.4 adelanta a la lluvia")
axs[0].set_ylabel("r")
axs[0].set_title("CCF Niño 3.4 vs anomalía de lluvia pampeana (todos los meses)")
n_prim = primavera.sum()
axs[1].bar(list(r_prim), list(r_prim.values()), color=COLOR["nino"], width=0.6)
axs[1].axhline(1.96 / np.sqrt(n_prim), color=COLOR["tinta2"], lw=0.8, ls="--")
axs[1].axhline(0, color=COLOR["tinta"], lw=0.8)
axs[1].set_xlabel("lag (meses)")
axs[1].set_title("Solo lluvia de oct–nov–dic")
fig.savefig(FIG / "06a_ccf_lluvia.png")
plt.show()
print(cc.set_index("lag").loc[-3:6, ["r", "umbral95"]].round(3).T.to_string())
print("primavera:", {k: round(v, 3) for k, v in r_prim.items()})

# %%
def granger(df: pd.DataFrame, causa: str, efecto: str, maxlag: int = 6) -> pd.Series:
    res = grangercausalitytests(df[[efecto, causa]], maxlag=maxlag)
    return pd.Series({lag: r[0]["ssr_ftest"][1] for lag, r in res.items()}, name=f"{causa} → {efecto}")


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    g = pd.concat([granger(m, "n34_anom", "precip_z"), granger(m, "precip_z", "n34_anom")], axis=1)
g.index.name = "lags"
print("p-valores del test F de Granger:")
print(g.map(lambda v: f"{v:.2e}" if v < 0.001 else f"{v:.3f}").to_string())

# %% [markdown]
# ### Segundo eslabón: lluvia de la ventana crítica → rendimiento (campañas 1981/82–2024/25)

# %%
lluvia_ventana = {"soja": (12, 3), "maiz": (11, 2), "trigo": (8, 11)}   # (mes inicio, mes fin)
MES = dict(enumerate(["", "ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]))


def lluvia_campania(anio: int, ini: int, fin: int) -> float:
    a0 = pd.Timestamp(anio, ini, 1)
    a1 = pd.Timestamp(anio + (1 if fin < ini else 0), fin, 1)
    s = clima.loc[a0:a1, "precip_mm"]
    return s.sum() if len(s) == (a1.year - a0.year) * 12 + a1.month - a0.month + 1 else np.nan


filas = []
for c, (ini, fin) in lluvia_ventana.items():
    for anio, a in A[c].dropna().items():
        filas.append({"cultivo": c, "campania": anio, "anom": a, "lluvia": lluvia_campania(anio, ini, fin),
                      "oni_ventana": oni_camp.loc[anio, VENTANA[c]].mean()})
ll = pd.DataFrame(filas).dropna()

fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, c in zip(axs, Y):
    d = ll[ll.cultivo == c]
    colores = d.oni_ventana.map(lambda v: col_f[fase_campania(v)])
    ax.scatter(d.lluvia, d.anom, c=colores, s=26, edgecolor=COLOR["fondo"], lw=0.8, zorder=3)
    rho = stats.spearmanr(d.lluvia, d.anom)
    b = np.polyfit(d.lluvia, d.anom, 1)
    xx = np.linspace(d.lluvia.min(), d.lluvia.max(), 10)
    ax.plot(xx, np.polyval(b, xx), color=COLOR["tinta2"], lw=1)
    ini, fin = lluvia_ventana[c]
    ax.set_title(f"{CULTIVO_NOMBRE[c]} · lluvia {MES[ini]}–{MES[fin]}\nρ = {rho.statistic:+.2f} ({fmt_p(rho.pvalue)})", fontsize=10)
    ax.set_xlabel("lluvia acumulada (mm)")
    ax.axhline(0, color=COLOR["tinta"], lw=0.8)
axs[0].set_ylabel("anomalía de rendimiento (%)")
for fz in orden_f:
    axs[0].scatter([], [], color=col_f[fz], s=26, label=fz)
axs[0].legend(fontsize=8, loc="lower right")
fig.suptitle("Lluvia en la ventana crítica vs anomalía de rendimiento (color: fase ENSO de la campaña)",
             x=0.01, ha="left", fontweight="semibold")
fig.savefig(FIG / "06b_lluvia_rendimiento.png")
plt.show()
print(ll.groupby("cultivo").apply(lambda d: pd.Series({
    "n": len(d),
    "rho(lluvia, anom)": stats.spearmanr(d.lluvia, d.anom).statistic,
    "rho(ONI, lluvia)": stats.spearmanr(d.oni_ventana, d.lluvia).statistic,
    "rho(ONI, anom)": stats.spearmanr(d.oni_ventana, d.anom).statistic,
}), include_groups=False).round(3).to_string())

# %% [markdown]
# ## 7. Resumen de la Etapa 1 (números de esta corrida)
#
# **ENSO como señal**
# 1. El ONI es un MA3 del Niño 3.4 mensual (r = 0,977). Último dato: **JAS 2026 = +2,16 °C**, con 4
#    trimestres seguidos ≥ +0,5: El Niño fuerte en formación de cara a la campaña 2026/27.
# 2. El 64 % de la varianza del Niño 3.4 está en la banda de 2 a 7 años, con picos en ~3,6 y ~5 años
#    que superan el ruido rojo AR(1) al 95 %. En la DWT, D4 + D5 (1,3 a 5,3 años) concentran el 66 % de la energía.
# 3. El escalograma CWT muestra que **ENSO no es estacionario en frecuencia**: hay mucha potencia
#    en los años 70 a 90 y menos en los 2000. Esto justifica usar métodos tiempo-frecuencia (Clase 10).
#
# **Rendimientos**
# 4. En nivel, las tres series son no estacionarias (ADF + KPSS). La anomalía relativa a una tendencia LOWESS
#    es estacionaria en los tres cultivos y casi no cambia con el filtro elegido (r ≥ 0,90 frente a MA9 y lineal).
#    El maíz no es estacionario ni siquiera alrededor de una tendencia lineal, porque su tendencia se aplana desde 2015.
#
# **ENSO → rendimiento (primera evidencia)**
# 5. Soja y maíz: ρ ≈ +0,42 a +0,51 con el ONI de **cualquier trimestre entre MJJ (invierno previo
#    a la siembra) y JFM de cosecha** (umbral 95 % = 0,26). La señal se apaga en MAM y AMJ de cosecha, cuando
#    el cultivo ya está definido, lo que es físicamente coherente. **Trigo: sin relación** (|ρ| < 0,16).
# 6. Según la fase de la campaña, la mediana de la anomalía de soja es −6,8 % en Niña y +7,2 % en Niño
#    (Kruskal-Wallis p = 0,002). En maíz es −6,4 % contra +8,6 % (p < 0,001). En años Niña, el 38 % de las campañas
#    pierde más del 10 %; en años Niño, el 10 %. Cuatro de las cinco peores campañas de soja fueron Niña
#    (2022/23 −44 %, 2008/09 −33 %, 1971/72, 1988/89).
#
# **Mecanismo mensual**
# 7. La CCF entre Niño 3.4 y la lluvia pampeana tiene su máximo en lag 0 (r = 0,21) y es mucho más fuerte
#    en primavera (oct a dic, r ≈ 0,37 con lags de 0 a 2). La lluvia de la ventana crítica explica la anomalía de soja (ρ = 0,71) y de maíz (0,54),
#    pero no la de trigo (0,10).
# 8. Granger Niño 3.4 → lluvia da p < 10⁻⁴. **El test inverso (lluvia → Niño 3.4) también rechaza H0 con lags de 2 a 6**,
#    y eso es físicamente imposible. Es la advertencia de la Clase 8: Granger mide información predictiva, no causalidad.
#    Hay factores comunes (la persistencia del ENSO, otros forzantes como el IOD o el SAM) y la teleconexión es
#    estacional (heterocedástica). Hay que controlarlo en la Etapa 2.
#
# **Implicancias para el diseño**: (i) el test de Granger tiene sentido a escala **mensual** o con el
# ONI medido **antes de la siembra**, y no "ONI de DEFM → rinde del mismo año"; (ii) el trigo sirve
# como control negativo; (iii) la campaña 2026/27 es un caso de pronóstico real fuera de muestra.
