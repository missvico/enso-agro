"""Etapa 1 - Preprocesamiento: de data/raw a data/processed.

Salidas (data/processed/):
- enso_mensual.parquet       ONI (centrado en el mes medio del trimestre), Niño 3.4 mensual, fase ENSO,
                             SAM (desde 1979), IOD (DMI HadISST), SOI estandarizado y
                             WWV (anomalía de volumen de agua cálida, 10^14 m³, desde 1980).
- clima_pampeano_mensual.parquet  precipitación (mm/mes) y temperatura promedio de 11 localidades
                                  pampeanas + anomalías respecto de la climatología mensual.
- rendimientos_anuales.parquet    rendimiento por campaña (nacional y región pampeana) para soja,
                                  maíz y trigo, sobre superficie cosechada y sembrada.
- campanias.parquet          una fila por cultivo-campaña: rendimiento + ONI de cada trimestre
                             desde el invierno previo a la siembra hasta el otoño de cosecha.

Convención de tiempo: `campania` = año de siembra (2022 = campaña 2022/23).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / "data" / "raw"
PROC = RAIZ / "data" / "processed"

TRIMESTRES = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
PROV_PAMPEANAS = {"06": "Buenos Aires", "14": "Córdoba", "82": "Santa Fe", "30": "Entre Ríos", "42": "La Pampa"}
CULTIVOS = {
    "soja": ("soja-serie-1941-2024-anual.csv", "soja-serie-1941-2024.csv"),
    "maiz": ("maiz-serie-1923-2024-anual.csv", "maiz-serie-1923-2024.csv"),
    "trigo": ("trigo-serie-1923-2025-anual.csv", "trigo-serie-1927-2025.csv"),
}


# ---------------------------------------------------------------- ENSO
def leer_enso() -> pd.DataFrame:
    oni = pd.read_csv(RAW / "noaa" / "oni.ascii.txt", sep=r"\s+")
    # Cada trimestre móvil se asigna a su mes central (DJF -> enero, NDJ -> diciembre).
    oni["mes"] = oni["SEAS"].map({s: i + 1 for i, s in enumerate(TRIMESTRES)})
    oni["fecha"] = pd.to_datetime(dict(year=oni["YR"], month=oni["mes"], day=1))
    oni = oni.set_index("fecha")[["ANOM"]].rename(columns={"ANOM": "oni"})

    n34 = pd.read_csv(RAW / "noaa" / "nino34_mensual.txt", sep=r"\s+")
    n34.columns = ["YR", "MON", "n12", "n12_anom", "n3", "n3_anom", "n4", "n4_anom", "n34", "n34_anom"]
    n34["fecha"] = pd.to_datetime(dict(year=n34["YR"], month=n34["MON"], day=1))
    n34 = n34.set_index("fecha")[["n34", "n34_anom"]]

    sam = pd.read_csv(RAW / "noaa" / "sam_aao_mensual.txt", sep=r"\s+", header=None, names=["YR", "MON", "sam"])
    sam.index = pd.to_datetime(dict(year=sam["YR"], month=sam["MON"], day=1))

    filas = []
    for linea in (RAW / "noaa" / "iod_dmi_mensual.txt").read_text().splitlines()[1:]:
        partes = linea.split()
        if len(partes) == 13 and partes[0].isdigit():
            filas += [(pd.Timestamp(int(partes[0]), m + 1, 1), float(v)) for m, v in enumerate(partes[1:])]
    iod = pd.DataFrame(filas, columns=["fecha", "iod"]).set_index("fecha").replace(-9999.0, np.nan)

    lineas = (RAW / "noaa" / "soi_mensual.txt").read_text().splitlines()
    ini = next(i for i, l in enumerate(lineas) if "STANDARDIZED" in l)
    filas = []
    for linea in lineas[ini:]:
        # formato de ancho fijo: año (4) + 12 columnas de 6 caracteres (los -999.9 vienen pegados)
        if len(linea) >= 76 and linea[:4].isdigit():
            for mm in range(12):
                filas.append((pd.Timestamp(int(linea[:4]), mm + 1, 1), float(linea[4 + 6 * mm:10 + 6 * mm])))
    soi = pd.DataFrame(filas, columns=["fecha", "soi"]).set_index("fecha").replace(-999.9, np.nan)

    wwv = pd.read_csv(RAW / "noaa" / "wwv_mensual.dat", sep=r"\s+", header=None, names=["d", "vol", "anom"],
                      skiprows=1, engine="python", on_bad_lines="skip")
    wwv = wwv[wwv["d"].astype(str).str.fullmatch(r"\d{6}")]
    wwv.index = pd.to_datetime(wwv["d"].astype(str), format="%Y%m")
    wwv = (wwv["anom"].astype(float) / 1e14).rename("wwv").to_frame()   # anomalía en 10^14 m³

    enso = (oni.join(n34, how="outer").join(sam[["sam"]], how="left").join(iod, how="left")
            .join(soi, how="left").join(wwv, how="left").asfreq("MS"))
    enso["fase"] = clasificar_fase(enso["oni"])
    return enso


def clasificar_fase(oni: pd.Series, umbral: float = 0.5, minimo: int = 5) -> pd.Series:
    """Definición operativa de NOAA: >= 5 trimestres consecutivos con |ONI| >= 0.5."""
    fase = pd.Series("Neutral", index=oni.index, dtype=object)
    for etiqueta, mascara in (("Niño", oni >= umbral), ("Niña", oni <= -umbral)):
        racha = (mascara != mascara.shift()).cumsum()
        largo = mascara.groupby(racha).transform("size")
        fase[mascara & (largo >= minimo)] = etiqueta
    fase[oni.isna()] = np.nan
    return fase


# ---------------------------------------------------------------- clima
def leer_clima() -> pd.DataFrame:
    series = []
    for f in sorted((RAW / "clima").glob("power_*.json")):
        p = json.loads(f.read_text())["properties"]["parameter"]
        df = pd.DataFrame(p)
        df = df[~df.index.str.endswith("13")]  # el mes "13" es el total anual
        df.index = pd.to_datetime(df.index, format="%Y%m")
        df = df.replace(-999, np.nan)
        df["precip_mm"] = df["PRECTOTCORR"] * df.index.days_in_month  # mm/día -> mm/mes
        df["localidad"] = f.stem.removeprefix("power_")
        series.append(df[["localidad", "precip_mm", "T2M", "T2M_MAX"]])
    largo = pd.concat(series)
    clima = largo.groupby(level=0)[["precip_mm", "T2M", "T2M_MAX"]].mean()
    clima.index.name = "fecha"
    clima = clima.asfreq("MS")
    for col in ["precip_mm", "T2M", "T2M_MAX"]:
        clim = clima[col].groupby(clima.index.month).transform("mean")
        clima[f"{col}_anom"] = clima[col] - clim
    # anomalía estandarizada de precipitación (la varianza cambia mucho entre meses)
    sd = clima["precip_mm"].groupby(clima.index.month).transform("std")
    clima["precip_z"] = clima["precip_mm_anom"] / sd
    return clima, largo


# ---------------------------------------------------------------- rendimientos
def leer_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, dtype={"provincia_id": str, "departamento_id": str})
        except UnicodeDecodeError:
            continue
    raise ValueError(path)


def leer_rendimientos() -> tuple[pd.DataFrame, pd.DataFrame]:
    filas, deptos = [], []
    for cultivo, (f_nac, f_dep) in CULTIVOS.items():
        nac = pd.read_csv(RAW / "magyp" / f_nac)
        nac.columns = ["campania", "sembrada_ha", "cosechada_ha", "produccion_t", "rend_kgha"]
        nac["region"] = "nacional"

        dep = leer_csv(RAW / "magyp" / f_dep)
        dep["cultivo"] = cultivo
        deptos.append(dep)
        pam = (
            dep[dep["provincia_id"].isin(PROV_PAMPEANAS)]
            .groupby("anio")[["superficie_sembrada_ha", "superficie_cosechada_ha", "produccion_tm"]]
            .sum()
            .reset_index()
        )
        pam.columns = ["campania", "sembrada_ha", "cosechada_ha", "produccion_t"]
        pam["rend_kgha"] = 1000 * pam["produccion_t"] / pam["cosechada_ha"]
        pam["region"] = "pampeana"

        for df in (nac, pam):
            df["cultivo"] = cultivo
            # rendimiento sobre superficie sembrada: incorpora el área perdida (sequías)
            df["rend_sembrada_kgha"] = 1000 * df["produccion_t"] / df["sembrada_ha"]
            df["perdida_area_pct"] = 100 * (1 - df["cosechada_ha"] / df["sembrada_ha"])
            filas.append(df)
    rend = pd.concat(filas, ignore_index=True)
    rend = rend[["cultivo", "region", "campania", "sembrada_ha", "cosechada_ha", "produccion_t",
                 "rend_kgha", "rend_sembrada_kgha", "perdida_area_pct"]]
    return rend, pd.concat(deptos, ignore_index=True)


# ---------------------------------------------------------------- tabla por campaña
def oni_por_campania(enso: pd.DataFrame) -> pd.DataFrame:
    """ONI de cada trimestre desde MJJ del año de siembra hasta AMJ del año de cosecha.

    Columnas `oni_<TRI>_<0|1>`: 0 = año de siembra, 1 = año siguiente.
    """
    oni = enso["oni"]
    filas = {}
    for anio in range(oni.index.year.min(), oni.index.year.max() + 1):
        fila = {}
        for desfase, meses in ((0, range(5, 13)), (1, range(1, 6))):
            for m in meses:
                fecha = pd.Timestamp(anio + desfase, m, 1)
                fila[f"oni_{TRIMESTRES[m - 1]}_{desfase}"] = oni.get(fecha, np.nan)
        filas[anio] = fila
    return pd.DataFrame.from_dict(filas, orient="index").rename_axis("campania")


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    enso = leer_enso()
    clima, clima_loc = leer_clima()
    rend, deptos = leer_rendimientos()
    camp = rend.merge(oni_por_campania(enso).reset_index(), on="campania", how="left")

    enso.to_parquet(PROC / "enso_mensual.parquet")
    clima.to_parquet(PROC / "clima_pampeano_mensual.parquet")
    clima_loc.to_parquet(PROC / "clima_localidades_mensual.parquet")
    rend.to_parquet(PROC / "rendimientos_anuales.parquet", index=False)
    deptos.to_parquet(PROC / "rendimientos_departamentos.parquet", index=False)
    camp.to_parquet(PROC / "campanias.parquet", index=False)

    print(f"ENSO: {enso.index.min():%Y-%m} a {enso['oni'].last_valid_index():%Y-%m}")
    print(f"Clima pampeano: {clima.index.min():%Y-%m} a {clima.index.max():%Y-%m}")
    print(rend.groupby(["cultivo", "region"])["campania"].agg(["min", "max", "count"]))


if __name__ == "__main__":
    main()
