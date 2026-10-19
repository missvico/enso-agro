# ENSO y rendimientos agrícolas pampeanos — TP Series Temporales (MEDyDC, FCEyN-UBA)

El plan completo (hipótesis, diseño, etapas, resultados y la revisión de la propuesta anterior) está en [docs/PLAN_FINAL.md](docs/PLAN_FINAL.md).

## Cómo correrlo

```bash
python3 -m venv .venv            # Python ≥ 3.10
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/00_descarga_datos.py      # baja todo a data/raw/ (≈ 5 MB)
.venv/bin/python src/01_preprocesamiento.py    # genera data/processed/*.parquet
cd notebooks && for nb in 01_exploratorio 02_inferencia 03_pronostico_enso 04_pronostico_rindes; do
  ../.venv/bin/jupytext --to ipynb $nb.py -o $nb.ipynb -q
  ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace $nb.ipynb; done
# figuras con la estética del deck:
cd .. && for nb in notebooks/0*.py; do TEMA_FIG=salvia FIG_DIR=figures/deck .venv/bin/python $nb; done
```

Los notebooks se editan como `.py` en formato *percent* (jupytext), porque así los diffs en git se pueden leer.
El `.ipynb` se regenera.

## Estructura

```
src/00_descarga_datos.py     descarga reproducible (NOAA, MAGyP, NASA POWER)
src/01_preprocesamiento.py   raw -> processed (parquet)
src/utils_ts.py              estilo de gráficos, ADF+KPSS (tabla Clase 11), ccf, tendencias
notebooks/01_exploratorio.*  Etapa 1: ONI, espectro, wavelets, rendimientos, ENSO↔rinde, Granger mensual
notebooks/02_inferencia.*    Etapa 2: Granger condicional (SAM/IOD/SOI/WWV), CCF preblanqueada, H1, panel, mediación
notebooks/03_pronostico_enso.*   Etapa 3a: STL, ARIMA, MLP, CNN 1D, LSTM, Prophet sobre Niño 3.4 (origen móvil)
notebooks/04_pronostico_rindes.* Etapa 3b: ARIMA, ARIMAX, Prophet e híbridos ARIMA + ML sobre rindes; 2026/27
results/                     métricas y pronósticos (json/csv)
figures/                     PNG (tema claro); figures/deck/ = mismas figuras con el tema del deck
assets/fonts/                tipografías del deck (para matplotlib con TEMA_FIG=salvia)
reports/                     notebook exportado a HTML
data/raw/                    fuentes originales (no se editan)
data/processed/              fuente de verdad para los análisis
```

## Datos

| Archivo procesado | Contenido | Fuente |
|---|---|---|
| `enso_mensual.parquet` | ONI, Niño 3.4 mensual, fase ENSO, SAM, IOD | NOAA CPC / PSL |
| `clima_pampeano_mensual.parquet` | lluvia (mm/mes), T media y máx., anomalías; promedio de 11 localidades | NASA POWER (1981–) |
| `rendimientos_anuales.parquet` | soja/maíz/trigo, nacional y pampeana, rinde s/cosechada y s/sembrada | MAGyP |
| `rendimientos_departamentos.parquet` | panel por departamento (para la extensión espacial) | MAGyP |
| `campanias.parquet` | una fila por cultivo-campaña + ONI de cada trimestre MJJ…AMJ⁺ | derivado |

Convención: `campania` = año de siembra (2022 = campaña 2022/23).
