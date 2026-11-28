# El Niño anticipa el rendimiento agrícola pampeano antes de la siembra
TP Data Mining de Series Temporales · MEDyDC, FCEyN-UBA
Victoria, Guillermo, Ramiro y Facundo San Martino · Noviembre 2026

*[Insertar gráfico: figures/deck/01_oni.png] · layout: portada minimalista con gráfico de fondo*

---

# ¿El estado del ENSO predice los rendimientos pampeanos, y por qué canal?
Tres hipótesis evaluadas con contraste exploratorio en trigo y pronóstico fuera de muestra

- **H1 (Inferencia):** El ONI previo a la siembra (JAS) predice la anomalía de soja y maíz (H0: β = 0).
- **H2 (Mecanismo):** Niño 3.4 Granger-causa la lluvia pampeana mensual y esta media el efecto al rinde.
- **H3 (Pronóstico):** Incluir ONI JAS mejora la precisión fuera de muestra frente al mejor modelo univariado.
- **Alcance adicional:** Trigo como contraste (exploratorio) y proyección real fuera de muestra para la campaña 2026/27.

*[Insertar gráfico: figures/deck/05a_perfil_correlacion.png] · layout: texto izquierda + gráfico derecha*

---

# Tres fuentes abiertas y un diseño estricto contra el data leakage temporal
Convención campania = año de siembra (2022 = 2022/23), ventanas fenológicas y horizonte causal

| Dimensión | Decisión metodológica | Por qué evita errores previos |
|---|---|---|
| Fuentes | NOAA (ONI, N3.4, WWV, SAM, IOD), MAGyP (1970–2024), NASA POWER (1981–2025) | Reproducible y sin huecos |
| Ventanas | Soja/Maíz: OND–DJF (crítico) / JAS (siembra); Trigo: JAS–SON / AMJ | Respeta el ciclo fenológico |
| Detrending | LOWESS para EDA/H1; log(rinde) con ARIMA(p,1,q)+drift en pronóstico | Sin fuga de años futuros |
| Causalidad | Granger mensual + confusores (descartamos CausalImpact) | Sin intervención discreta |

*[Insertar gráfico: figures/deck/04a_rendimientos.png] · layout: tabla arriba + gráfico abajo*

---

# El Niño oscila cada 2 a 7 años y concentra allí el 64 % de su varianza
El ONI es un filtro MA3 centrado de la anomalía mensual Niño 3.4 (r = 0,977)

- Periodograma suavizado (N = 918, 1950–2026): **63,7 %** de la varianza entre **2 y 7 años**.
- Picos significativos (p < 0,05 vs ruido rojo AR(1), r₁ = 0,953) en **3,64 años** y **4,50–5,46 años**.
- Descomposición DWT (db4): los detalles D4 y D5 (1,3–5,3 años) concentran el **66,4 %** de la energía.
- Señal actual: **4 trimestres seguidos ≥ +0,5 °C**, alcanzando ONI JAS 2026 = +2,16 °C.

*[Insertar gráfico: figures/deck/02_espectro_nino34.png] · layout: gráfico izquierda + texto derecha*

---

# La señal ENSO y su acople con la lluvia pampeana son intermitentes
Escalograma CWT Morlet y coherencia wavelet Niño 3.4 ↔ lluvia (1981–2025)

| Década | Potencia CWT Niño 3.4 (2–7a) | Coherencia media R² (COI) | Área sig. 95 % (fase media) |
|---|---:|---:|---:|
| 1980s | 4,78 | 0,463 | 0,0 % (sin coherencia) |
| 1990s | 4,54 | 0,474 | 23,7 % (−19,7°) |
| 2000s | 2,60 | 0,534 (máx. 0,968) | 36,0 % (+27,3°) |
| 2010s–2020s | 3,08 (2020s) | 0,341 | 2,8 % (< 5 %, sin coherencia) |

- Acoplamiento significativo concentrado en 1990–2009 casi en fase (media circular +10,0°, 0,174 rad, R = 0,875, n = 1.649 celdas); 1980s y 2010–2025 no superan el umbral del azar AR(1) (5 %).

*[Insertar gráfico: figures/deck/20_coherencia_wavelet.png] · layout: tabla izquierda + mapa wavelet derecha*

---

# La anomalía de rinde es estacionaria y responde al ENSO en cultivos de verano
Tests ADF+KPSS (n = 55, 1970–2024) y separación de campañas según fase fenológica

- Niveles de rinde son no estacionarios (p_KPSS = 0,010); maíz falla incluso con tendencia lineal (meseta post-2015).
- Las anomalías LOWESS (%) y Δlog son estacionarias en los tres cultivos (p_ADF < 0,001, p_KPSS ≥ 0,056).
- **Soja:** mediana **−6,8 %** (Niña) vs **+7,2 %** (Niño), Kruskal-Wallis p = 0,002.
- **Maíz:** mediana **−6,4 %** (Niña) vs **+8,6 %** (Niño), p < 0,001; **Trigo:** sin señal (p = 0,239).

*[Insertar gráfico: figures/deck/05b_fases.png] · layout: texto izquierda + gráfico derecha*

---

# El calor subsuperficial (WWV) explica la falsa causalidad inversa en Granger
VAR mensual (1981–2025, n = 540) y función de correlación cruzada preblanqueada

- En VAR bivariado, Niño 3.4 → lluvia (p = 0,0003) convive con un imposible lluvia → Niño 3.4 (p = 0,0174).
- La señal inversa resiste SAM, IOD, SOI, errores HC3 (p = 0,013) y permutación por bloques (p = 0,015).
- Al controlar por el calor subsuperficial (WWV → Niño 3.4, p = 3,5e-18), **el inverso desaparece (p = 0,130)**.
- Preblanqueada por AR(p), la CCF se concentra en **lag 0 (r = 0,154)**: teleconexión intra-mensual.

*[Insertar gráficos: figures/deck/07a_granger_condicional.png + figures/deck/07_ccf_preblanqueada.png] · layout: texto arriba + dos gráficos abajo*

---

# H1 confirmada: el ONI pre-siembra predice soja y maíz en toda la región
Modelos anidados (n = 54), panel departamental agrupado por año y mediación (n = 44)

- **Soja (H1):** β = +9,10 %/°C (IC95: 3,53–14,67), p_Holm = 0,0019; R² sube de 0,020 a 0,191.
- **Maíz (H1):** β = +9,32 %/°C (IC95: 4,51–14,13), p_Holm = 0,0006; R² sube de 0,012 a 0,238.
- **Panel departamental:** **98,3 %** de 116 deptos (soja) y **100 %** de 128 deptos (maíz) positivos (p_cluster ≈ 10⁻⁴).
- **Mediación por lluvia:** **63,1 %** del efecto en soja y **50,0 %** en maíz pasa por precipitación crítica.

*[Insertar gráfico: figures/deck/08_panel_departamentos.png] · layout: KPIs izquierda + histogramas derecha*

---

# La pendiente puntual duplica en La Niña, pero el test formal no rechaza linealidad
Modelo quebrado anom ~ y₁ + ONI⁺ + ONI⁻ (1971–2024, errores HC3 y 2.000 bootstraps; residuos parciales)

| Cultivo | β_Niña (ONI⁻) | β_Niño (ONI⁺) | Test Wald HC3 (H0: β⁻ = β⁺) |
|---|---:|---:|---:|
| Soja | +12,83 %/°C (p = 0,016) | +6,33 %/°C (p = 0,092) | p = 0,382 (boot 0,398) |
| Maíz | +14,61 %/°C (p = 0,018) | +5,56 %/°C (p = 0,085) | p = 0,255 (boot 0,261) |

- Con n = 54 años, ARIMAX asimétrico rolling no supera al lineal (421,9 vs 409,7 kg/ha en soja).

*[Insertar gráfico: figures/deck/17_asimetria_nina_nino.png] · layout: tabla arriba + gráfico abajo*

---

# Para pronosticar Niño 3.4, el ensamble ARIMA + MLP domina de 3 a 9 meses
Evaluación con origen móvil (2000–2026, 318 orígenes); Prophet fracasa sin memoria autorregresiva

- **Corto plazo (h = 1):** ARIMA(2,0,2) (reajustado cada año) lidera con RMSE = 0,198 °C (ACC = 0,971).
- **Mediano plazo (h = 6):** MLP alcanza RMSE = 0,668 °C (ACC = 0,613) frente a 0,758 °C (0,435) del ARIMA.
- **Ensamble ARIMA + MLP:** supera significativamente al ARIMA en h = 3, 6, 9 (Diebold-Mariano p < 0,01).
- **Contraejemplo (Prophet):** ACC negativo en todos los horizontes (−0,04 a −0,34).

*[Insertar gráfico: figures/deck/12_skill_modelos_enso.png] · layout: gráfico izquierda + texto derecha*

---

# La barrera de predictibilidad favorece al agro y el 2026 ya supera los pronósticos
Skill a 6 meses según mes de origen y seguimiento en vivo del Niño 2026/27

- **Barrera de otoño austral (h = 6):** desde enero ACC cae a −0,01 (ARIMA) y +0,29 (MLP).
- **Ventana de siembra:** desde julio–septiembre ACC sube a 0,66–0,75 (ARIMA) y 0,77–0,83 (MLP).
- **Verificación real JAS 2026:** observado +2,16 °C vs pronosticado desde junio (MLP +1,72, ARIMA+MLP +1,63).
- Para **DJF 2026/27**, el ensamble proyecta +1,52 °C (MLP +2,16 °C): Niño fuerte en período crítico.

*[Insertar gráficos: figures/deck/13_barrera_predictibilidad.png + figures/deck/14_pronostico_2026.png] · layout: dos gráficos en paralelo + texto abajo*

---

# El ONI pre-siembra reduce el error fuera de muestra en maíz y soja, no en trigo
Origen móvil causal (1995–2024, n = 30) sobre log(rinde) frente a ARIMA(p,1,q) + drift

- **Maíz:** ARIMAX reduce el RMSE de 1.123,7 a 951,6 kg/ha (+28,3 % skill, p_DM = 0,043).
- **Soja:** ARIMAX baja de 449,9 a 409,7 kg/ha (+17,1 % skill, p_DM = 0,097).
- **Árboles híbridos (4 vars):** GBM llega a 840,7 en maíz (+44,0 %) y RF a 403,7 en soja (+19,5 %), destacando en Niñas.
- **Trigo (contraste exploratorio):** ARIMAX empeora (488,9 vs 466,9 kg/ha, −9,7 % skill).

*[Insertar gráfico: figures/deck/15_rmse_modelos_rindes.png] · layout: texto izquierda + gráfico derecha*

---

# Ablación de baselines y puente 3a → 3b desagregan tendencia y señal ENSO
Separando tendencia por tramos (Prophet sin ONI), variables extra (solo ONI) y Niño 3.4 pronosticado

- **Prophet sin ONI:** en trigo ya mejora +19,5 % (419,0 vs 466,9); en maíz aporta +14,1 % por quiebre 2015, y sumar ONI baja otros 136,6 kg/ha (p_kg = 0,043).
- **RF/GBM (solo ONI, 5 semillas):** pierden ventaja en soja (454,2 / 478,8) y caen debajo de ARIMAX en maíz; parte del éxito híbrido venía de SOI, IOD y tendencia del ONI (trigo AMJ sin mejora sig., p ≥ 0,145).
- **Trigo causal (ONI AMJ):** no aparece señal ENSO (477,7 kg/ha, −4,7 % skill, p_DM = 0,850; MA con θ ≈ −1 en el borde de invertibilidad).
- **Puente 3a → 3b (soja y maíz, r = 0,909):** ARIMAX con Niño 3.4 DJF pronosticado (ARIMA(2,0,2)) logra +17,6 % en soja (p = 0,046) y +19,2 % en maíz (p = 0,035); trigo excluido por cosecha nov–dic.

*[Insertar gráfico: figures/deck/18_baselines_y_puente_3a_3b.png] · layout: gráfico izquierda + texto derecha*

---

# Diagnósticos de residuos, horizonte h = 2 y test Diebold-Mariano + Holm
Comparación primaria M3 vs M2 en escala log (soja y maíz); kg/ha y demás modelos como sensibilidad/exploratorios

| Modelo vs M2 ARIMA | Soja p_DM log (kg/ha) | Maíz p_DM log (kg/ha) | Holm log (Soja / Maíz) | Holm sens. kg/ha (Maíz) |
|---|---:|---:|---:|---:|
| M3 ARIMAX (ONI JAS) | 0,097 (0,115) | 0,043 (0,025) | 0,097 / 0,087 | 0,050 |
| M3 Puente (N3.4 DJF hat, expl.) | 0,046 (0,044) | 0,035 (0,020) | 0,069 / 0,069 | 0,040 (Soja 0,044) |
| Híbrido RF (4 vars, expl.) | 0,188 (0,231) | 0,047 (0,057) | 0,188 / 0,094 | 0,115 |

- Ljung-Box (lags = 5, 10; model_df = 1) **no rechaza ruido blanco** en M2 ni M3 (p ≥ 0,121). En h = 2 (n = 29), ARIMAX retiene +14,3 % (soja) y +23,4 % (maíz) con cobertura IC95 = 93,1 %.

*[Insertar gráfico: figures/deck/19_diagnosticos_y_dm.png] · layout: tabla arriba + gráfico abajo*

---

# Campaña 2026/27: proyección con ONI = +2,16 °C y alerta por extrapolación
Comparativa entre ARIMA sin ENSO, ARIMAX lineal y ARIMAX asimétrico (modelo de sensibilidad)

| Cultivo | ARIMA (sin ENSO) | ARIMAX lineal (IC80) | ARIMAX asimétrico (IC80) |
|---|---:|---:|---:|
| Soja 2026/27 | 3.153 kg/ha | **3.887 kg/ha** (3.184–4.746) | 3.584 kg/ha (2.944–4.362) |
| Maíz 2026/27 | 7.654 kg/ha | **9.096 kg/ha** (7.562–10.940) | 8.532 kg/ha (7.116–10.230) |

- **Advertencia crítica:** +2,16 °C excede el máximo histórico 1970–2024 (+1,79 °C en 1997); en Niños extremos puede haber excesos hídricos no capturados por una pendiente lineal.

*[Insertar gráfico: figures/deck/16_pronosticos_rindes.png] · layout: tabla izquierda + gráfico derecha*

---

# H1 y H2 se sostienen plenamente; H3 queda parcialmente sostenida
Síntesis de evidencia, limitaciones metodológicas honestas y próximos pasos

- **Veredicto:** El ONI pre-siembra predice soja y maíz (H1), mediado 50–63 % por lluvia con WWV como confusor océano-atmósfera (H2). En pronóstico (H3), el beneficio es consistente en maíz y en el puente 3a→3b sin ajuste múltiple, quedando parcialmente sostenido tras Holm en escala log (p_Holm = 0,087 y 0,097); en trigo no aparece señal.
- **Límites asumidos:** muestra efectiva de 55 campañas; θ ≈ −1 en ARIMA(0,1,1) de trigo; redes de Etapa 3a congeladas en 1999 (y early_stopping aleatorio en MLPRegressor); IOD faltante en 2026 para híbridos.
- **Cierre:** Pipeline modular reproducible (00–05 + tests), verificación inmediata con publicación MAGyP 2026/27.

*[Sin gráfico · layout: 3 bloques/tarjetas de cierre (Veredicto, Límites, Próximos pasos)]*
