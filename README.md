# ENSO y rendimientos agrícolas pampeanos — TP Series Temporales (MEDyDC, FCEyN-UBA)

El plan completo (hipótesis, diseño, etapas, resultados y la revisión de la propuesta anterior) está en [docs/PLAN_FINAL.md](docs/PLAN_FINAL.md). La guía para armar las diapositivas en Gamma y el guion narrado están en [docs/DECK_GAMMA.md](docs/DECK_GAMMA.md) y [docs/GUION_VIDEO.md](docs/GUION_VIDEO.md).

## Cómo correrlo

### Linux / macOS
```bash
python3 -m venv .venv            # Python ≥ 3.10
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/00_descarga_datos.py      # baja todo a data/raw/ (≈ 5,6 MB; --actualizar-noaa para refrescar NOAA)
.venv/bin/python src/01_preprocesamiento.py    # genera data/processed/*.parquet
cd notebooks && for nb in 01_exploratorio 02_inferencia 03_pronostico_enso 04_pronostico_rindes 05_extensiones_analiticas; do
  ../.venv/bin/jupytext --to ipynb $nb.py -o $nb.ipynb -q
  ../.venv/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=900 --inplace $nb.ipynb
  ../.venv/bin/jupyter nbconvert --to html $nb.ipynb --output-dir ../reports; done
# figuras con la estética del deck:
cd .. && for nb in notebooks/0*.py; do TEMA_FIG=salvia FIG_DIR=figures/deck .venv/bin/python $nb; done
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe src\00_descarga_datos.py
.venv\Scripts\python.exe src\01_preprocesamiento.py
foreach ($nb in "01_exploratorio","02_inferencia","03_pronostico_enso","04_pronostico_rindes","05_extensiones_analiticas") {
  .venv\Scripts\jupytext.exe --to ipynb "notebooks\$nb.py" -o "notebooks\$nb.ipynb" -q
  .venv\Scripts\python.exe -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=900 --inplace "notebooks\$nb.ipynb"
  .venv\Scripts\python.exe -m nbconvert --to html "notebooks\$nb.ipynb" --output-dir reports
}
# figuras con la estética del deck:
$env:MPLBACKEND="Agg"; $env:TEMA_FIG="salvia"; $env:FIG_DIR="figures/deck"
Get-ChildItem notebooks\0*.py | ForEach-Object { .venv\Scripts\python.exe $_.FullName }
```

Los notebooks se editan como `.py` en formato *percent* (jupytext), porque así los diffs en git se pueden leer.
El `.ipynb` se regenera.

## Estructura

```
src/00_descarga_datos.py           descarga reproducible (NOAA, MAGyP, NASA POWER; --actualizar-noaa opcional)
src/01_preprocesamiento.py         raw -> processed (parquet)
src/utils_ts.py                    estilo de gráficos, ADF+KPSS (tabla Clase 11), ccf, tendencias, dm_test (Bartlett)
notebooks/01_exploratorio.*        Etapa 1: ONI, espectro, wavelets, rendimientos, ENSO↔rinde, Granger mensual
notebooks/02_inferencia.*          Etapa 2: Granger condicional (SAM/IOD/SOI/WWV), CCF preblanqueada, H1, panel, mediación
notebooks/03_pronostico_enso.*     Etapa 3a: STL, ARIMA, MLP, CNN 1D, LSTM, Prophet sobre Niño 3.4 (origen móvil)
notebooks/04_pronostico_rindes.*   Etapa 3b: ARIMA, ARIMAX, Prophet e híbridos ARIMA + ML sobre rindes; 2026/27
notebooks/05_extensiones_analiticas.* Etapa 4: baselines desambiguados, asimetría Niña/Niño, puente 3a→3b, h=2, DM+Holm, coherencia wavelet
docs/PLAN_FINAL.md                 plan metodológico v2
docs/DECK_GAMMA.md                 16 slides listas para importar en Gamma
docs/GUION_VIDEO.md                guía de estilo Gamma + guion narrado (~15 min)
results/                           métricas y pronósticos (json/csv)
figures/                           PNG (tema claro); figures/deck/ = mismas figuras con el tema del deck
assets/fonts/                      tipografías del deck (para matplotlib con TEMA_FIG=salvia)
reports/                           notebook exportado a HTML
data/raw/                          fuentes originales (no se editan)
data/processed/                    fuente de verdad para los análisis
```

## Datos

| Archivo procesado | Contenido | Fuente |
|---|---|---|
| `enso_mensual.parquet` | ONI, Niño 3.4 mensual, fase ENSO, SAM, IOD, SOI, WWV | NOAA CPC / PSL / PMEL |
| `clima_pampeano_mensual.parquet` | lluvia (mm/mes), T media y máx., anomalías; promedio de 11 localidades | NASA POWER (1981–) |
| `clima_localidades_mensual.parquet` | serie mensual por localidad pampeana (11 puntos, 1981–2025) | NASA POWER (1981–) |
| `rendimientos_anuales.parquet` | soja/maíz/trigo, nacional y pampeana, rinde s/cosechada y s/sembrada | MAGyP |
| `rendimientos_departamentos.parquet` | panel por departamento (para la extensión espacial) | MAGyP |
| `campanias.parquet` | una fila por cultivo-campaña + ONI de cada trimestre AMJ…AMJ⁺ | derivado |

Convención: `campania` = año de siembra (2022 = campaña 2022/23).
