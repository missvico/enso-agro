"""Descarga reproducible de todas las fuentes de datos del TP.

Uso (desde la raíz del proyecto):
    .venv/bin/python src/00_descarga_datos.py

Fuentes:
- NOAA CPC: índice ONI (trimestres móviles) y Niño 3.4 mensual (ERSSTv5, base 1991-2020).
- NOAA CPC / PSL / PMEL: SAM, IOD, SOI y Warm Water Volume (WWV), posibles confusores.
- MAGyP (datos.magyp.gob.ar): estimaciones agrícolas de soja, maíz y trigo,
  serie nacional anual y serie por departamento.
- NASA POWER (power.larc.nasa.gov): precipitación y temperatura mensual (1981-hoy)
  en localidades de la Región Pampeana. Reemplaza al SMN, que pide registro.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / "data" / "raw"

NOAA = {
    "oni.ascii.txt": "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
    "nino34_mensual.txt": "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii",
    # posibles confusores / forzantes adicionales de la lluvia pampeana (Etapa 2)
    "sam_aao_mensual.txt": "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/aao/monthly.aao.index.b79.current.ascii",
    "iod_dmi_mensual.txt": "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data",
    "soi_mensual.txt": "https://www.cpc.ncep.noaa.gov/data/indices/soi",
    # contenido de calor subsuperficial del Pacífico ecuatorial (precursor del ENSO)
    "wwv_mensual.dat": "https://www.pmel.noaa.gov/tao/wwv/data/wwv.dat",
}

_MAGYP = "https://datos.magyp.gob.ar/dataset"
MAGYP = {
    "soja-serie-1941-2024-anual.csv": f"{_MAGYP}/8ae4865f-d2f2-45a2-9343-7a4a12728a90/resource/3bfe4630-91e2-46cd-82ad-cf8dc94d3dfb/download/soja-serie-1941-2024-anual.csv",
    "soja-serie-1941-2024.csv": f"{_MAGYP}/8ae4865f-d2f2-45a2-9343-7a4a12728a90/resource/ba694aa3-99d2-4d7d-9936-60f88e36ad9a/download/soja-serie-1941-2024.csv",
    "maiz-serie-1923-2024-anual.csv": f"{_MAGYP}/514853fc-0a78-4b6f-a42f-8e89eab784c8/resource/c4696e28-4e54-464b-bc3b-19c608371231/download/maiz-serie-1923-2024-anual.csv",
    "maiz-serie-1923-2024.csv": f"{_MAGYP}/514853fc-0a78-4b6f-a42f-8e89eab784c8/resource/9a6a02f8-ef58-4250-87c2-f639fec502f1/download/maiz-serie-1923-2024.csv",
    "trigo-serie-1923-2025-anual.csv": f"{_MAGYP}/10105e94-c560-4b02-b15f-ef3ef764b833/resource/dc3ae069-af81-4bdd-82cb-2f57f2c296dd/download/trigo-serie-1923-2025-anual.csv",
    "trigo-serie-1927-2025.csv": f"{_MAGYP}/10105e94-c560-4b02-b15f-ef3ef764b833/resource/50f0edcc-4dfc-4afb-b78a-b164601d36ae/download/trigo-serie-1927-2025.csv",
}

# Localidades de la zona núcleo y su periferia (lat, lon).
LOCALIDADES = {
    "pergamino": (-33.89, -60.57),
    "junin": (-34.58, -60.95),
    "pehuajo": (-35.81, -61.90),
    "tres_arroyos": (-38.38, -60.28),
    "marcos_juarez": (-32.70, -62.10),
    "rio_cuarto": (-33.12, -64.35),
    "laboulaye": (-34.13, -63.39),
    "rafaela": (-31.25, -61.49),
    "casilda": (-33.04, -61.17),
    "parana": (-31.73, -60.53),
    "general_pico": (-35.66, -63.76),
}
POWER_URL = (
    "https://power.larc.nasa.gov/api/temporal/monthly/point"
    "?parameters=PRECTOTCORR,T2M,T2M_MAX&community=AG"
    "&longitude={lon}&latitude={lat}&start=1981&end={fin}&format=JSON"
)


def bajar(url: str, destino: Path, forzar: bool = False) -> None:
    if destino.exists() and not forzar:
        print(f"  ya existe {destino.relative_to(RAIZ)}")
        return
    destino.parent.mkdir(parents=True, exist_ok=True)
    print(f"  bajando {destino.relative_to(RAIZ)}")
    with urllib.request.urlopen(url, timeout=300) as r:
        destino.write_bytes(r.read())


def main(forzar: bool = False) -> None:
    print("NOAA")
    for nombre, url in NOAA.items():
        bajar(url, RAW / "noaa" / nombre, forzar=True)  # se actualiza mensualmente

    print("MAGyP")
    for nombre, url in MAGYP.items():
        bajar(url, RAW / "magyp" / nombre, forzar)

    print("NASA POWER")
    fin = time.localtime().tm_year - 1  # último año completo
    for nombre, (lat, lon) in LOCALIDADES.items():
        destino = RAW / "clima" / f"power_{nombre}.json"
        bajar(POWER_URL.format(lat=lat, lon=lon, fin=fin), destino, forzar)
        json.loads(destino.read_text())  # valida que la respuesta sea JSON
        time.sleep(1)


if __name__ == "__main__":
    import sys

    main(forzar="--forzar" in sys.argv)
