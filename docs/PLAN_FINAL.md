# PLAN FINAL — ENSO y rendimientos agrícolas pampeanos

**TP Data Mining de Series Temporales — MEDyDC, FCEyN-UBA** · Versión 2 (18/10/2026)
Reemplaza a `plan_proyecto.md` (v1) y a la propuesta de `series-temporales-enso-agro-main/`.
Código y datos: este repositorio. Resultados: `notebooks/` y `reports/`.

---

## 1. Revisión de la propuesta anterior (PPT + `series-temporales-enso-agro-main`)

### Qué se rescata (está bien planteado)
- **Tema y fuentes:** el ONI de NOAA y las estimaciones del MAGyP son las fuentes correctas, oficiales y largas.
- **Estructura del video** alineada con la consigna, y la narrativa "un método de la cursada por resultado".
- **Split temporal**, nunca aleatorio. Es correcto.
- **ADF y KPSS combinados**, con la tabla de decisión de la Clase 11.
- **ARIMAX como baseline estadístico** y comparación con RMSE/MAE.
- **Discusión de los límites de Granger** (confusores, correlación ≠ causalidad).
- Las **ventanas fenológicas** como idea para pasar de mensual a campaña.
- Ingeniería: una única fuente de verdad en parquet, estructura `data/raw` y `data/processed`, entorno virtual.

### Qué está mal y cómo se corrige

| # | Problema | Gravedad | Por qué está mal | Corrección |
|---|---|---|---|---|
| 1 | **Granger anual con el ONI de DEFM de la misma campaña** | Alta | `grangercausalitytests` usa solo lags ≥ 1. Con datos anuales, testea si el ONI **de la campaña anterior** predice el rinde de esta. Esa no es la hipótesis. El efecto es intra-campaña (lag 0), y lag 0 no entra en Granger. | Hacer Granger donde tiene sentido: (a) a escala **mensual**, Niño 3.4 → lluvia pampeana (540 meses); (b) a escala anual, con el **ONI medido antes de la siembra** (JAS), que es legítimamente "pasado" respecto del rinde, mediante un test F de modelos anidados. |
| 2 | **CCF anual "para hallar el lag óptimo"** | Alta | Con una serie anual, los lags son años y el relevante es 0. No hay lag que buscar. | Barrer el **trimestre** en que se mide el ONI (MJJ…AMJ⁺) contra el rinde: es un "perfil de correlación". Ya está hecho en la Etapa 1. |
| 3 | **Una sola ventana (dic a mar) para todos los cultivos** | Alta | El trigo se siembra en jun/jul y se cosecha en nov/dic. DEFM cae **después** de la cosecha. | Ventanas por cultivo: soja y maíz OND–NDJ–DJF; trigo JAS–ASO–SON. |
| 4 | **LightGBM + Optuna + TimeSeriesSplit + SHAP con unas 50 campañas** | Alta | Con unos 45 años de train y folds internos de unos 10 puntos, el sobreajuste está garantizado y el SHAP es inestable. Además, los **árboles no extrapolan la tendencia tecnológica**: desde 2018 los rindes quedan fuera del rango de entrenamiento y el modelo predice "planchado". Tampoco es un método de la materia (la Clase 12 ve CNN, LSTM y Prophet). | Modelar la **anomalía** (sin tendencia) con ARIMAX / regresión con errores ARMA. Si se quiere un modelo de la Clase 12, usar **Prophet con regresor**. LightGBM queda como sensibilidad opcional, sobre la anomalía y con advertencia de n. |
| 5 | **Hold-out único desde 2018** (unas 7 campañas) | Media | Con 7 puntos, una diferencia de RMSE no tiene poder estadístico. La PPT dice "superará **estadísticamente**", pero no define ningún test. | **Origen móvil** (expanding window, 1 paso) sobre 1995/96–2024/25, unos 30 pronósticos, más el **test de Diebold-Mariano**. |
| 6 | **Leakage temporal** | Media | El ONI de DEFM recién se conoce al final del período crítico. Una tendencia centrada (MA o LOWESS) usa años futuros. | En el pronóstico, usar solo lo disponible en la siembra (ONI JAS) y una tendencia **causal**. El ONI de DEFM queda como escenario "exógena perfecta" (cota superior). |
| 7 | **Se omite media materia** | Media | La consigna pide aplicar lo visto en clase. Faltan FFT/espectro (Cl. 3), filtros (4, 5, 9), wavelets (10), interpolación (7) y contrafáctico (8). | Se incorporan. Espectro, DWT, CWT, filtros y ccf ya están en la Etapa 1. |
| 8 | **Comparaciones múltiples** | Media | Barrer lags, ventanas y cultivos y quedarse con el máximo infla los falsos positivos. | Ventana principal fijada **a priori** por fenología. El resto se reporta como sensibilidad. Corrección de Holm entre cultivos. |
| 9 | **SMN como fuente de lluvia** | Baja | Pide registro y las series de estación tienen huecos. | **NASA POWER** (API sin registro, mensual 1981–2025, 11 localidades pampeanas). Ya está descargado. |
| 10 | **"Limpieza de nulos e interpolación"** de rindes | Baja | La serie pampeana 1970–2024 no tiene huecos. Interpolar rindes inventaría datos. | No interpolar. La interpolación (Cl. 7) se usa solo donde corresponde (por ejemplo, pasar de mensual a campaña con filtrado previo). |
| 11 | **Rinde nacional sobre superficie cosechada** | Baja | El agregado nacional mezcla la expansión a zonas marginales. Además, el rinde sobre cosechada esconde el área perdida en sequías (soja 2022/23: 11,5 % de área no cosechada). | La serie principal es la **Región Pampeana** (BA, Cba, SF, ER, LP). Como robustez, el rinde **sobre sembrada** (solo soja y trigo; en maíz no sirve porque una parte va a forraje) y el panel departamental. |
| 12 | "El clima es el principal factor de variabilidad" | Baja | Es una afirmación sin cita. | Citar (p. ej. Podestá et al. 1999) o reformular. |

**Veredicto:** el tema y el esqueleto son buenos. El núcleo metodológico (Granger anual con DEFM, LightGBM con unas 50
observaciones, una sola ventana y un hold-out de 7 puntos) **no resistiría la defensa** y hay que reemplazarlo. La Etapa 1
ya muestra que la versión corregida tiene resultados claros.

---

## 2. Pregunta, hipótesis y objetivos

**Pregunta:** ¿el estado del ENSO aporta información predictiva sobre el rendimiento de los cultivos pampeanos, y
por qué mecanismo?

| | Hipótesis (H0 entre paréntesis) | Test | Clase |
|---|---|---|---|
| **H1 (principal)** | El ONI observado **antes de la siembra** (JAS) aporta información predictiva sobre la anomalía de rendimiento de **soja y maíz** pampeanos, más allá de la historia propia del rendimiento, con signo positivo (Niño ↑, Niña ↓). *(H0: β_ONI = 0)* | F de modelos anidados (estilo Granger), p-valor por permutación, Holm entre cultivos | 8, 11 |
| **H2 (mecanismo)** | El Niño 3.4 causa en el sentido de Granger la anomalía de lluvia pampeana a escala mensual, y la lluvia de la ventana crítica media el efecto ONI → rinde. *(H0: Niño 3.4 no Granger-causa la lluvia)* | Granger en VAR mensual (incondicional y condicional a SAM e IOD), CCF preblanqueada, análisis de mediación simple | 6, 8 |
| **H3 (pronóstico)** | Un modelo con el ONI pre-siembra como exógena reduce el error de pronóstico fuera de muestra respecto del mejor modelo univariado. *(H0: igual precisión)* | Origen móvil 1995/96–2024/25, RMSE/MAE/skill, Diebold-Mariano | 11, 12 |

- **Contraste exploratorio (trigo):** en la Etapa 1 el trigo no mostró relación con el ONI. Se reporta como contraste, aclarando que
  surgió de mirar los datos y que no fue una hipótesis previa.
- **Caso de aplicación:** el ONI de JAS 2026 es +2,16 °C, un Niño fuerte en formación. Se hará un pronóstico explícito para la campaña
  **2026/27**, verificable cuando el MAGyP publique los datos.

**Objetivos específicos:**
1. Caracterizar el ENSO en tiempo, frecuencia y tiempo-frecuencia.
2. Estacionarizar los rindes (quitar la tendencia tecnológica con filtros pasa-bajos) y validarlo con ADF + KPSS.
3. Cuantificar la relación ENSO → lluvia → rinde (ccf, Granger, perfil por trimestre).
4. Comparar pronósticos con y sin ENSO con una evaluación honesta.
5. Discutir los límites: confusores, n chico, no estacionariedad del ENSO y cambio climático.

---

## 3. Datos (todos ya descargados en `enso-agro/data/raw/`)

| Serie | Fuente | Frecuencia / período | Cómo se obtiene |
|---|---|---|---|
| ONI | NOAA CPC `oni.ascii.txt` | mensual (trimestre móvil), 1950–ago 2026 | `src/00_descarga_datos.py` |
| Niño 3.4 (sin suavizar) | NOAA CPC ERSSTv5 (base 1991–2020) | mensual, 1950–jun 2026 | idem |
| SAM (AAO) e IOD (DMI) | NOAA CPC / PSL | mensual, 1979– / 1870– | idem (posibles confusores) |
| Rinde soja, maíz y trigo | MAGyP, datos.magyp.gob.ar | anual por campaña; nacional y **departamental** (1923/1941–2024/25) | idem (CSV abiertos, sin registro) |
| Lluvia y temperatura | NASA POWER (MERRA-2/IMERG) | mensual, 1981–2025, 11 localidades | idem (API sin registro) |

Datos opcionales si sobra tiempo: lluvia más larga (CRU TS o GPCC, desde 1950) a través de KNMI Climate Explorer. Se baja como
CSV, se guarda en `data/raw/clima/` y se agrega un lector en `01_preprocesamiento.py`.

---

## 4. Metodología por etapas

### Etapa 0: infraestructura ✅ (hecha)
Descarga reproducible, preprocesamiento a parquet, entorno, skill `series-temporales-expert`.

### Etapa 1: análisis exploratorio ✅ (hecha; se presenta el 19/10)
| Análisis | Método | Clase | Resultado clave |
|---|---|---|---|
| ONI = MA3 del Niño 3.4 | filtro MA | 4 | r = 0,977 |
| Espectro del Niño 3.4 | FFT (k·Fs/N), Hanning, AR(14), ruido rojo, Parseval | 3, 9, 11 | 64 % de la varianza entre 2 y 7 años; picos en 3,6 y 5 años (p < 0,05) |
| DWT / CWT | Mallat db4, Morlet + COI | 10 | D4 + D5 = 66 % de la energía; potencia ENSO no estacionaria (alta en 1970–90) |
| Tendencia de rindes | lineal / MA9 / LOWESS | 4, 11 | anomalías robustas al filtro (r ≥ 0,90) |
| Estacionariedad | ADF + KPSS | 11 | nivel no estacionario → anomalía estacionaria |
| Perfil ONI × rinde | ρ de Spearman por trimestre | 6 | soja y maíz ρ ≈ 0,45–0,51 **desde el invierno previo a la siembra**; trigo ≈ 0 |
| Fases ENSO | Kruskal-Wallis, Mann-Whitney | — | soja −6,8 % (Niña) contra +7,2 % (Niño), p = 0,002 |
| ENSO → lluvia | CCF mensual, Granger | 6, 8 | r = 0,21 (0,37 en primavera); Granger p < 10⁻⁴, **pero el inverso también da significativo** |

### Etapa 2: inferencia formal (H1 y H2) ✅ (hecha; `notebooks/02_inferencia`)
| Análisis | Resultado |
|---|---|
| Granger en VAR mensual, directo | Niño 3.4 → lluvia con p ≤ 0,005 en todas las especificaciones |
| Granger inverso (lluvia → Niño 3.4) | resiste a SAM, IOD, SOI y al placebo por permutación (p = 0,015); **desaparece al controlar por el calor subsuperficial (WWV)**: p = 0,10–0,17. Ese es el confusor de la Clase 8 |
| CCF preblanqueada | solo lag 0 (r = 0,15): la teleconexión actúa dentro del mes |
| **H1** (ONI jul–sep → anomalía) | soja +9,1 %/°C (p = 0,002), maíz +9,3 %/°C (p = 0,0003), los dos significativos con Holm; trigo 0 (p = 0,95). Robusto en 8 variantes |
| Panel departamental | 98 % (soja) y 100 % (maíz) de los departamentos con ρ > 0; p ≈ 10⁻⁴ agrupando por campaña (sin agrupar daría 10⁻⁶²) |
| Mediación por la lluvia | 63 % (soja) y 50 % (maíz) del efecto |

### Etapa 3: modelado y pronóstico (Clases 11 y 12) ✅ (hecha; `notebooks/03` y `04`)
**3a. Niño 3.4 mensual** (train 1950–1999, test 2000–2026 con origen móvil y horizontes de 1 a 9 meses):
- STL: tendencia de +0,4 °C. Por ACF/PACF y AICc queda un ARIMA(2,0,2) (un oscilador amortiguado), con residuos blancos.
- Modelos: ARIMA, MLP, CNN 1D, LSTM y Prophet.
  - El ARIMA gana a 1 y 2 meses.
  - El **MLP** gana a 4–9 meses (ACC 0,61 contra 0,44 a 6 meses).
  - El **ensamble ARIMA + MLP** le gana al ARIMA con DM p < 0,01 en h = 3, 6 y 9.
  - Prophet falla (ACC < 0): no tiene memoria autorregresiva.
- Barrera de predictibilidad: a 6 meses, ACC de unos 0,0 desde enero (el pronóstico cruza el otoño) contra unos 0,7 desde julio.
- Caso real 2026: todos los modelos subestimaron el JAS observado (+2,16). Para DJF, el ensamble da +1,5.

**3b. Rindes** (origen móvil 1995/96–2024/25, 30 pronósticos):
- ARIMA(0,1,1) con drift como baseline.
- **Maíz:** el ARIMAX con ONI jul–sep tiene un skill de +28 % (DM p = 0,043). El híbrido ARIMA + GBM llega a +44 %.
- **Soja:** ARIMAX +17 % (p = 0,097). El híbrido ARIMA + RF da +20 % y es el mejor en años Niña.
- **Trigo:** sin mejora (contraste).
- El oráculo con el ONI de dic–feb no supera al de jul–sep.
- Con Holm, H3 queda **parcialmente** sostenida.
- **Pronóstico 2026/27:** soja ≈ 3.890 kg/ha (IC80 3.180–4.750) y maíz ≈ 9.100 kg/ha (IC80 7.560–10.940). Se extrapola, porque +2,16 está fuera del rango histórico.

**Pendientes:**
- Coherencia wavelet entre el ENSO y la lluvia.
- Usar el pronóstico del Niño 3.4 como exógena del modelo de rindes.
- Probar un efecto asimétrico Niña/Niño (los árboles lo sugieren).
- Sensibilidad de los híbridos con WWV, que existe desde 1980 y por eso tiene un n más chico.

### Etapa 4: discusión, video y entrega
- Límites: n ≈ 55, confusores (SAM, IOD, cambio tecnológico ligado a años específicos), no estacionariedad del ENSO,
  cambio climático, lluvia de reanálisis contra estaciones, agregación espacial.
- Por qué **no** usar CausalImpact: no hay una intervención discreta con un período "post" y un control independiente de X.
  Explicarlo muestra que se entiende la Clase 8.
- Bibliografía y anexo de código.

---

## 5. Guion del video (≈ 15 min)

| Min | Sección | Contenido / figura |
|---|---|---|
| 0–1 | Carátula + gancho | El Niño 2026 en curso (fig. 01) |
| 1–3 | Introducción | ENSO, teleconexión con la Pampa, sequía 2022/23 |
| 3–4 | Hipótesis y objetivos | H1–H3 (tabla de la sección 2) |
| 4–6 | Material y métodos | fuentes, ventanas fenológicas, pipeline, métodos por clase |
| 6–11 | Resultados | espectro y wavelets (02, 03) → rindes y estacionariedad (04) → perfil y fases (05) → Granger mensual (06) → H1 → pronóstico |
| 11–14 | Discusión | Granger inverso, confusores, límites, 2026/27 |
| 14–15 | Conclusiones + bibliografía | |

---

## 6. Cronograma y roles (ajustar a la fecha real de entrega)

| Semana | Tarea | Rol sugerido |
|---|---|---|
| S0 (19/10) | Presentación de avance de las Etapas 1 a 3 | todos |
| S1 | Pendientes de modelado (coherencia wavelet, ENSO pronosticado como insumo, efecto asimétrico) | persona A |
| S1 | Revisión de resultados y redacción de métodos | persona B |
| S2 | Figuras finales y guion del video | A + B |
| S3 | Grabación | persona C |
| S4 | Revisión y entrega | todos |

**Pendiente de definir con el equipo:** fecha de entrega, integrantes y roles, y si se suma Prophet o LightGBM como sensibilidad.

---

## 7. Bibliografía base
- Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 37, 424–438.
- Brodersen, K. H. et al. (2015). Inferring causal impact using Bayesian structural time-series models. *Ann. Appl. Stat.*, 9(1), 247–274.
- Torrence, C. & Compo, G. P. (1998). A practical guide to wavelet analysis. *BAMS*, 79(1), 61–78.
- Taylor, S. J. & Letham, B. (2018). Forecasting at scale. *The American Statistician*, 72(1), 37–45.
- Podestá, G. P. et al. (1999). Associations between grain crop yields in central-eastern Argentina and El Niño–Southern Oscillation. *J. Appl. Meteorol.*, 38, 1488–1498.
- Grimm, A. M., Barros, V. R. & Doyle, M. E. (2000). Climate variability in southern South America associated with El Niño and La Niña events. *J. Climate*, 13, 35–58.
- Diebold, F. X. & Mariano, R. S. (1995). Comparing predictive accuracy. *J. Bus. Econ. Stat.*, 13(3), 253–263.
- NOAA CPC — Oceanic Niño Index (ONI v5). MAGyP — Estimaciones agrícolas. NASA POWER — Prediction of Worldwide Energy Resources.
