# Guía de Armado en Gamma y Guion del Video (`15 min`)

---

## 0. Guía rápida para configurar Gamma (1 minuto)

### Paleta exacta del tema `salvia` (extraída de `src/utils_ts.py`)
Para que las diapositivas en Gamma coincidan exactamente con el fondo y tipografía de las figuras en `figures/deck/`:

| Elemento | Valor exacto | Uso en Gamma |
|---|---|---|
| **Fondo de tarjeta / slide** | `#E8F0E2` | Background principal (`COLOR["fondo"]`) |
| **Texto principal y títulos** | `#0B1A15` | Títulos y texto cuerpo (`COLOR["tinta"]`) |
| **Texto secundario / subtítulos** | `#3F5248` | Bajadas, fuentes y notas al pie (`COLOR["tinta2"]`) |
| **Bordes, divisores y grillas de tabla** | `#D2DFD0` | Líneas suaves (`COLOR["grilla"]`) |
| **Gris neutro** | `#A3B2A6` | Elementos secundarios (`COLOR["neutral"]`) |
| **Acento primario (cálido / tendencia)** | `#C4622D` | Resaltados principales y badges (`COLOR["tendencia"]`) |
| **Acento Niño / alerta** | `#e34948` | Valores Niño y umbrales críticos |
| **Acento Soja / serie** | `#2a78d6` | Soja, curvas principales y fase Niña |
| **Acento Maíz** | `#eb6834` | Maíz |
| **Acento Trigo** | `#1baf7a` | Trigo / control negativo |

### Tipografías (`assets/fonts/`)
- **Títulos (Display):** `Instrument Serif` (`InstrumentSerif-Regular.ttf`)
- **Cuerpo y subtítulos:** `Familjen Grotesk` (`FamiljenGrotesk-wght.ttf`)
- **Código, métricas y ejes numéricos:** `Space Mono` (`SpaceMono-Regular.ttf`)

### Pasos para importar en Gamma y exportar `.pptx`
1. Abrí Gamma → **Crear nuevo con IA** → **Pegar texto** (*Paste in text*).
2. Abrí `docs/DECK_GAMMA.md`, seleccioná todo (`Ctrl+A`) y pegalo en Gamma.
3. En las opciones de importación elegí:
   - **Modo de texto:** *Conservar exactamente este texto* (*Preserve this exact text*).
   - **División de tarjetas:** *Por divisores `---`* (*By `---` card dividers*).
   - **Tema:** Creá un tema personalizado con fondo `#E8F0E2`, texto `#0B1A15`, acento `#C4622D` y fuentes `Instrument Serif` + `Familjen Grotesk` (o las más cercanas disponibles en Gamma si no subís los `.ttf` de `assets/fonts/`).
4. En cada una de las 16 tarjetas generadas, reemplazá la línea en cursiva `*[Insertar gráfico: figures/deck/...]*` subiendo el PNG correspondiente desde `figures/deck/` y aplicá el layout sugerido.
   - *Nota:* Si querés incluir dos figuras en las Slides 4, 5, 6, 7, 10, 11 o 12, podés sumar el segundo gráfico mencionado en el subtítulo/contenido (`01_oni.png`, `03b_cwt_nino34.png`, `04a_rendimientos.png`, `07_ccf_preblanqueada.png`, `11_residuos_arima.png`, `14_pronostico_2026.png`, `16_pronosticos_rindes.png`) como grilla de 2 imágenes.
5. Exportá como **PowerPoint (`.pptx`)** desde `Compartir → Exportar`.

---

## 1. Guion narrado (~15 minutos · Slides 1 a 16)

### Slide 1 · `00:00–00:45` · Speaker: **Vico**
**Texto:**
> Hola a todos. Hoy presentamos nuestro trabajo final de Data Mining de Series Temporales sobre la relación entre el fenómeno El Niño–Oscilación del Sur (ENSO) y los rendimientos agrícolas en la Región Pampeana argentina. A diferencia de los enfoques descriptivos tradicionales, este proyecto combina análisis espectral y tiempo-frecuencia, inferencia con control de confusores y evaluación estricta fuera de muestra. Y tenemos un gancho en tiempo real: mientras cerrábamos el análisis, el índice ONI de julio–septiembre de 2026 escaló a +2,16 grados, configurando un evento El Niño fuerte en formación cuyo pronóstico para la campaña 2026/27 dejamos explícitamente documentado y verificable.

**Qué señalar en pantalla:** Curva roja creciente al extremo derecho del gráfico `01_oni.png` (+2,16 °C en 2026) y comparación visual con los súper Niños de 1982/83, 1997/98 y 2015/16.

---

### Slide 2 · `00:45–01:40` · Speaker: **Facundo**
**Texto:**
> La pregunta guía es concreta: ¿el estado del ENSO aporta información predictiva real sobre los rendimientos pampeanos al momento de decidir la siembra, y a través de qué mecanismo físico? Para responderla planteamos tres hipótesis formales. Primero, H1: el ONI observado antes de la siembra, en el trimestre julio–septiembre, predice la anomalía de rendimiento de soja y maíz más allá de su propia historia. Segundo, H2: la anomalía mensual de Niño 3.4 Granger-causa la lluvia pampeana, y esa precipitación durante la ventana crítica media el efecto sobre el rinde. Tercero, H3: incorporar ese ONI pre-siembra reduce el error de pronóstico fuera de muestra frente al mejor modelo univariado. Además incluimos el trigo como control negativo natural, por ser un cultivo de invierno cuya ventana crítica precede o diverge de la teleconexión estival.

**Qué señalar en pantalla:** El perfil de correlación por trimestre (`05a_perfil_correlacion.png`), mostrando que soja y maíz superan el umbral estadístico desde `MJJ/JAS`, mientras la curva verde del trigo permanece plana cerca de cero.

---

### Slide 3 · `01:40–02:45` · Speaker: **Vico**
**Texto:**
> Toda la ingeniería se diseñó para evitar fugas temporales. Integramos tres fuentes abiertas: índices oceánicos y atmosféricos de NOAA desde 1950, series agrícolas provinciales y departamentales del MAGyP, y precipitación grillada de NASA POWER para 11 localidades pampeanas desde 1981. Adoptamos la convención agronómica de indexar cada campaña por el año de siembra y definimos las ventanas fenológicas a priori para evitar búsqueda múltiple de rezagos. Un punto metodológico clave: en la etapa exploratoria usamos filtros LOWESS para visualizar anomalías, pero al pasar a pronóstico modelamos directamente el logaritmo del rendimiento con ARIMA más drift, evitando usar años futuros en la tendencia. También descartamos explícitamente `CausalImpact` porque el ENSO es un forzante continuo y cíclico sin intervención discreta ni grupo control independiente.

**Qué señalar en pantalla:** La tabla de decisiones anti-leakage y la curva de tendencia tecnológica no lineal del maíz en `04a_rendimientos.png`, que se aplana desde 2015.

---

### Slide 4 · `02:45–03:45` · Speaker: **Facundo**
**Texto:**
> Empezamos caracterizando la señal climática en tiempo y frecuencia. Primero verificamos que el ONI oficial es prácticamente un promedio móvil centrado de orden 3 sobre la anomalía mensual de Niño 3.4, con una correlación de 0,977; la pequeña diferencia residual obedece a que NOAA actualiza sus climatologías base de 30 años cada lustro. Al pasar al dominio de la frecuencia con la Transformada Rápida de Fourier sobre los 918 meses sin suavizar, vemos que la banda clásica del ENSO, entre 2 y 7 años, concentra el 63,7 % de toda la varianza. El periodograma suavizado con ventana de Hanning y el espectro autorregresivo muestran picos significativos por encima del espectro de ruido rojo AR(1) al 95 % alrededor de los 3,6 y 5,1 años. En wavelets discretas `db4`, los detalles D4 y D5 retienen el 66,4 % de la energía total.

**Qué señalar en pantalla:** La banda sombreada de 2 a 7 años en `02_espectro_nino34.png` y los picos que sobrepasan la línea punteada del ruido rojo AR(1).

---

### Slide 5 · `03:45–04:45` · Speaker: **Vico**
**Texto:**
> Sin embargo, Fourier promedia toda la historia y oculta cuándo cambia la intensidad del fenómeno. Al aplicar la Transformada Wavelet Continua de Morlet comprobamos que el ENSO no es estacionario en frecuencia: la potencia en la banda 2–7 años fue alta en los años 80 y 90 y decayó en los 2000. Para cerrar una brecha clave de la etapa exploratoria, implementamos la coherencia wavelet al cuadrado de Grinsted entre Niño 3.4 y la lluvia pampeana estandarizada entre 1981 y 2025, evaluando significancia con 250 simulaciones Monte Carlo AR(1). El mapa tiempo-período muestra un acoplamiento en fase significativo especialmente entre mediados de los 90 y 2010 —donde en los 2000 la coherencia media alcanza 0,534 y el 36 % del área supera el umbral del 95 %—, debilitándose notablemente en 2010–2025 hasta 0,341.

**Qué señalar en pantalla:** Los contornos negros significativos y las flechas horizontales hacia la derecha ("en fase") en la banda de 2–4 años entre 1992 y 2012 en `20_coherencia_wavelet.png`.

---

### Slide 6 · `04:45–05:45` · Speaker: **Facundo**
**Texto:**
> Pasando a los cultivos, trabajamos sobre la Región Pampeana desde 1970 hasta 2024, 55 campañas sin datos faltantes. Combinando los tests ADF y KPSS verificamos que las series en nivel son no estacionarias en los tres cultivos, y en maíz ni siquiera es estacionaria alrededor de una tendencia lineal debido a la meseta productiva post-2015. Tanto la tasa anual `Δlog` como la anomalía relativa respecto a LOWESS logran estacionariedad clara. Al clasificar las campañas por la fase ENSO de su ventana fenológica, la diferencia es contundente en verano: la mediana de soja cae a −6,8 % en años Niña frente a +7,2 % en años Niño, con p = 0,002; en maíz pasa de −6,4 % a +8,6 %, con p menor a 0,001; y en trigo no hay separación significativa.

**Qué señalar en pantalla:** Los puntos azules negativos en La Niña (como 2022/23 con −44 % en soja) frente a las medianas positivas en El Niño en `05b_fases.png`, contrastando con la dispersión plana del trigo.

---

### Slide 7 · `05:45–06:50` · Speaker: **Vico**
**Texto:**
> Para testear el mecanismo físico H2 trabajamos sobre los 540 meses desde 1981 hasta 2025. En un VAR bivariado mensual encontramos inicialmente un resultado paradójico: Niño 3.4 Granger-causa la lluvia pampeana con p = 0,0003, pero la dirección inversa —lluvia causando al Pacífico— también da significativa con p = 0,0174. Ese resultado resiste errores robustos HC3, permutaciones por bloques de 12 meses y el agregado de otros modos climáticos como el SAM, el IOD o el SOI. ¿Cuál era la variable omitida? Al incluir el volumen de agua cálida subsuperficial del Pacífico, el índice WWV —que predice al Niño 3.4 con p = 3,5×10⁻¹⁸—, la causalidad inversa desaparece por completo (p = 0,13 a 0,17), mientras el efecto directo sigue firme. Además, al preblanquear por un AR(p), la correlación cruzada queda concentrada en lag 0: la teleconexión actúa dentro del mismo mes.

**Qué señalar en pantalla:** En `07a_granger_condicional.png`, cómo en el panel derecho todas las barras inversas cruzan la línea `p = 0,05` hasta que aparece `+ WWV` y vuelven al gris no significativo.

---

### Slide 8 · `06:50–07:45` · Speaker: **Facundo**
**Texto:**
> Con el canal físico validado, testeamos H1 sobre la anomalía anual usando el ONI observado en julio–septiembre, antes de la siembra de verano. Rechazamos H0 para ambos cultivos tras corregir por Holm: cada grado Celsius adicional en ONI JAS suma +9,10 % en soja (p de Holm = 0,0019) y +9,32 % en maíz (p = 0,0006), elevando el R² de alrededor de 0,02 a 0,19 y 0,24 respectivamente, mientras el trigo permanece en cero. Para descartar que el resultado lo arrastren pocos departamentos, repetimos el análisis sobre el panel departamental agrupando errores estándar por campaña: 98,3 % de 116 partidos sojeros y el 100 % de 128 partidos maiceros muestran correlación positiva, evitando la ilusión de p-valores de 10⁻⁶² por pseudo-replicación. Por último, el análisis de mediación estima que el 63,1 % del efecto en soja y el 50,0 % en maíz viaja por la lluvia de la ventana crítica.

**Qué señalar en pantalla:** Los histogramas en `08_panel_departamentos.png` desplazados casi por completo a la derecha del cero, y la comparación entre `p agrupado` y `p ingenuo`.

---

### Slide 9 · `07:45–08:40` · Speaker: **Vico**
**Texto:**
> Una pregunta natural es si el daño de La Niña supera al beneficio de El Niño, como suelen sugerir modelos de árboles. Estimamos una regresión quebrada con pendientes separadas para ONI positivo y negativo usando errores HC3 y 2.000 réplicas bootstrap. Puntualmente la asimetría es llamativa: la pendiente en La Niña duplica o casi triplica a la de El Niño (+12,8 vs +6,3 %/°C en soja; +14,6 vs +5,6 %/°C en maíz). Sin embargo, con 54 campañas la incertidumbre es alta y el test formal de Wald no rechaza igualdad de pendientes (p = 0,382 en soja y 0,255 en maíz). Al llevar este `ARIMAX asimétrico` a pronóstico rolling, sumar ese parámetro extra no reduce el RMSE global frente al ARIMAX lineal, aunque sí cumple un rol clave como cota conservadora para escenarios de Niño extremo.

**Qué señalar en pantalla:** La pendiente más empinada a la izquierda del cero y la banda ancha del IC95% hacia la derecha del máximo histórico (`+1,79 °C`) en `17_asimetria_nina_nino.png`.

---

### Slide 10 · `08:40–09:40` · Speaker: **Facundo**
**Texto:**
> En la Etapa 3a evaluamos si podemos pronosticar la anomalía mensual de Niño 3.4 de 1 a 9 meses adelante usando un esquema de origen móvil desde enero de 2000 hasta junio de 2026 (318 orígenes). Ajustamos un `ARIMA(2,0,2)` sobre 1950–1999 —cuyo par autorregresivo complejo actúa como oscilador amortiguado y deja residuos blancos— y lo reajustamos cada año en ventana expansiva. En horizontes de 1 y 2 meses gana el ARIMA (RMSE = 0,198 °C y ACC = 0,971 en h=1). A partir de 4 a 9 meses toma la delantera el MLP entrenado sobre ventanas de 24 meses, logrando ACC = 0,613 en h=6 frente a 0,435 del ARIMA. El ensamble simple `ARIMA + MLP` supera significativamente al ARIMA con Diebold-Mariano p < 0,01 en h = 3, 6 y 9. En contraste, Prophet fracasa con ACC negativo porque carece de memoria autorregresiva para una oscilación no periódica anual.

**Qué señalar en pantalla:** El cruce de las curvas en `12_skill_modelos_enso.png` alrededor de `h = 3`, el umbral `ACC = 0,6` y la curva de Prophet por debajo de cero.

---

### Slide 11 · `09:40–10:35` · Speaker: **Vico**
**Texto:**
> Al desagregar la precisión del pronóstico según el mes calendario en que se emite, emerge claramente la llamada barrera de predictibilidad de otoño austral. A 6 meses vista, un pronóstico emitido en enero —que debe cruzar el otoño— cae a una correlación de −0,01 en ARIMA y 0,29 en MLP. En cambio, un pronóstico emitido entre julio y septiembre alcanza un ACC de 0,66 a 0,75 en ARIMA y de 0,77 a 0,83 en MLP. Por fortuna agronómica, la ventana de siembra de maíz y soja cae del lado favorable de esa barrera. En el caso real de 2026, todos los modelos entrenados hasta junio anticiparon la fase cálida aunque subestimaron la intensidad récord de `JAS` (+2,16 °C), siendo el MLP (+1,72 °C) y el ensamble (+1,63 °C) los más cercanos, proyectando un Niño fuerte para el verano 2026/27.

**Qué señalar en pantalla:** La banda clara horizontal en los meses de enero a marzo del mapa de calor `13_barrera_predictibilidad.png` comparada con la zona azul oscuro en julio–septiembre.

---

### Slide 12 · `10:35–11:30` · Speaker: **Facundo**
**Texto:**
> En la Etapa 3b evaluamos el pronóstico anual de rendimientos en ventana expansiva sobre 30 campañas (1995/96 a 2024/25) modelando `log(rinde)`. Agregar el ONI de julio–septiembre en un ARIMAX reduce el RMSE fuera de muestra del maíz en 28,3 % de skill (bajando de 1.124 a 952 kg/ha, Diebold-Mariano p = 0,043) y el de la soja en 17,1 % (de 450 a 410 kg/ha, p = 0,097). Los híbridos de ARIMA más árboles sobre residuos (`GBM` y `RF`) alcanzan hasta +44 % de skill en maíz y +19,5 % en soja, recortando drásticamente el error en años Niña (de 1.476 a 829 kg/ha en maíz). Como esperábamos, el trigo no mejora con el ENSO. Y un resultado notable: el modelo oráculo con ONI de diciembre–febrero no supera de manera sustancial al ONI pre-siembra.

**Qué señalar en pantalla:** Las barras rojas (`ARIMAX ONI JAS`) y azules (`Híbridos`) acortándose respecto de la barra negra (`ARIMA`) en Maíz y Soja en `15_rmse_modelos_rindes.png`, mientras en Trigo empeoran.

---

### Slide 13 · `11:30–12:35` · Speaker: **Vico**
**Texto:**
> Para asegurar honestidad absoluta en los baselines, agregamos cuatro controles críticos. Primero, evaluamos `Prophet sin ONI`: en trigo logra +19,5 % de skill frente a ARIMA —superando incluso a `Prophet + ONI`—, lo que prueba que el aparente +14,9 % de Prophet en trigo era pura flexibilidad de tendencia y cero señal ENSO. En maíz, la tendencia por tramos aporta +14,1 % por el quiebre de 2015, y sumar ONI aporta los restantes 136,6 kg/ha de reducción de RMSE (p_kg = 0,043). Segundo, restringiendo `RF` y `GBM` a usar únicamente `oni_jas`, ambos pierden ventaja en soja y caen debajo de `ARIMAX` en maíz, revelando que parte de su ganancia venía de sumar `SOI`, `IOD` y tendencia del ONI. Tercero, re-evaluamos el trigo con `ONI AMJ` estrictamente pre-siembra y se confirma su nulidad (−4,7 %). Cuarto, construimos el puente explícito entre 3a y 3b pronosticando `DJF` desde septiembre (`r = 0,909`): ese `M3_puente` logra +17,6 % de skill en soja y +19,2 % en maíz, ambos significativos.

**Qué señalar en pantalla:** Comparación visual en `18_baselines_y_puente_3a_3b.png` entre `Prophet sin ONI` vs `Prophet + ONI`, `RF/GBM solo ONI` vs `4 vars`, y la barra roja de `M3 Puente`.

---

### Slide 14 · `12:35–13:25` · Speaker: **Facundo**
**Texto:**
> También cerramos tres verificaciones estadísticas formales. Los tests de Ljung-Box y los correlogramas de `Δlog` verifican que los residuos de `M2` y `M3` se comportan como ruido blanco en los tres cultivos. Al probar el horizonte `h = 2` en ventana móvil —que replica la situación real de proyectar 2026/27 sin conocer aún el rinde 2025/26—, `ARIMAX` retiene +14,3 % de skill en soja y +23,4 % en maíz con una cobertura empírica del IC95 % del 93,1 %. Por último, unificamos el test de Diebold-Mariano en escala log y en kg/ha con corrección formal de Holm: mientras en soja el `ARIMAX` lineal queda en p = 0,097 (sosteniendo H3 de forma parcial en soja), tanto en maíz como usando el puente 3a→3b en pérdida en kg/ha los p-valores ajustados por Holm alcanzan significancia (0,040 y 0,044).

**Qué señalar en pantalla:** Los gráficos ACF/PACF de la fila superior y el umbral vertical `p = 0,05` en las barras de Diebold-Mariano en `19_diagnosticos_y_dm.png`.

---

### Slide 15 · `13:25–14:15` · Speaker: **Vico**
**Texto:**
> Aplicando todo el pipeline a la campaña en curso 2026/27 con un ONI observado en julio–septiembre de +2,16 °C, el `ARIMAX` lineal proyecta 3.887 kg/ha para soja (+23,3 % sobre el ARIMA sin ENSO) y 9.096 kg/ha para maíz (+18,8 %). Sin embargo, debemos ser sumamente cautelosos: +2,16 °C supera el máximo histórico de nuestra muestra de entrenamiento desde 1970, que fue +1,79 °C en 1997. Si usamos el modelo `ARIMAX asimétrico`, que toma en cuenta la menor pendiente marginal observada en años Niño, las proyecciones se moderan a 3.584 kg/ha en soja y 8.532 kg/ha en maíz. Además, en eventos El Niño extremos existen riesgos agronómicos por excesos hídricos y anegamientos que una relación lineal no captura.

**Qué señalar en pantalla:** Las diferencias entre las estimaciones en la tabla comparativa y la zona reciente de `16_pronosticos_rindes.png` con las franjas azules de campañas Niña pasadas.

---

### Slide 16 · `14:15–15:00` · Speaker: **Facundo**
**Texto:**
> En síntesis: H1 y H2 quedan sólidamente respaldadas —el ONI pre-siembra contiene información predictiva sobre soja y maíz mediada por la lluvia pampeana y desambiguada mediante el calor subsuperficial WWV—, mientras H3 se sostiene con claridad en maíz y en el puente 3a→3b. Asumimos con transparencia los límites del estudio: contamos con 55 campañas efectivas, en la Etapa 3a las redes neuronales mantuvieron pesos congelados desde 1999 con early stopping aleatorio en MLP frente a un ARIMA reentrenado, y el índice IOD faltante en 2026 impidió proyectar 2026/27 con los modelos híbridos de 4 variables. Todo el flujo es reproducible de punta a punta en el repositorio y la campaña 2026/27 nos dará la verificación definitiva apenas el MAGyP publique sus datos. Muchas gracias.

**Qué señalar en pantalla:** Las curvas de pronóstico e intervalos de confianza para 2026/27 en `14_pronostico_2026.png` y el resumen de veredicto H1–H3.
