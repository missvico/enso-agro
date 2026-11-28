"""Análisis tiempo-frecuencia con wavelet continua de Morlet y coherencia wavelet (Grinsted et al., 2004)."""
from __future__ import annotations

import os

import numpy as np


def grilla_morlet(
    n: int,
    dt: float = 1.0 / 12.0,
    dj: float = 1.0 / 12.0,
    omega0: float = 6.0,
    s0: float | None = None,
) -> dict:
    """Precomputa escalas, períodos, filtros Morlet en Fourier, kernels de suavizado y COI."""
    if n < 4:
        raise ValueError(f"La serie requiere al menos 4 muestras (n={n}).")
    flambda = (4.0 * np.pi) / (omega0 + np.sqrt(2.0 + omega0 ** 2))
    s0 = 2.0 * dt if s0 is None else float(s0)
    j_max = int(np.floor(np.log2(n * dt / s0) / dj))
    scales = s0 * 2.0 ** (np.arange(j_max + 1) * dj)
    periods = flambda * scales

    n_pad = int(2 ** np.ceil(np.log2(n * 2)))
    k_fft = np.arange(n_pad)
    omega = np.where(
        k_fft <= n_pad // 2,
        2.0 * np.pi * k_fft / (n_pad * dt),
        -2.0 * np.pi * (n_pad - k_fft) / (n_pad * dt),
    )

    so = np.outer(scales, omega)
    heaviside = (omega > 0.0)[None, :]
    norm_w = np.sqrt(2.0 * np.pi * scales / dt)[:, None] * (np.pi ** -0.25)
    psi_hat = norm_w * np.exp(-0.5 * (so - omega0) ** 2) * heaviside

    time_kernels = []
    for sc in scales:
        sig_samp = sc / dt
        half = min(int(np.ceil(3.0 * sig_samp)), n)
        tt = np.arange(-half, half + 1)
        kern = np.exp(-0.5 * (tt / sig_samp) ** 2)
        kern /= kern.sum()
        time_kernels.append(kern)

    n_sc_box = int(round(0.60 / dj))
    if n_sc_box % 2 == 0:
        n_sc_box += 1

    t_dist = (0.5 + np.minimum(np.arange(n), n - 1 - np.arange(n))) * dt
    coi_period = flambda * (t_dist / np.sqrt(2.0))
    inside_coi = periods[:, None] <= coi_period[None, :]

    return {
        "n": n,
        "dt": dt,
        "dj": dj,
        "omega0": omega0,
        "flambda": flambda,
        "scales": scales,
        "periods": periods,
        "n_pad": n_pad,
        "psi_hat": psi_hat,
        "time_kernels": time_kernels,
        "n_sc_box": n_sc_box,
        "coi_period": coi_period,
        "inside_coi": inside_coi,
    }


def cwt_morlet(arr: np.ndarray, grilla: dict) -> np.ndarray:
    """Transformada Wavelet Continua Morlet para serie (N,) o lote (B, N)."""
    arr = np.asarray(arr, dtype=float)
    a2d = np.atleast_2d(arr)
    f_arr = np.fft.fft(a2d, n=grilla["n_pad"], axis=-1)
    prod = f_arr[:, None, :] * grilla["psi_hat"][None, :, :]
    w = np.fft.ifft(prod, axis=-1)[..., :grilla["n"]]
    return w[0] if arr.ndim == 1 else w


def smooth_wt(W: np.ndarray, grilla: dict) -> np.ndarray:
    """Suavizado gaussiano en tiempo y boxcar en escala según Grinsted et al. (2004)."""
    orig_2d = W.ndim == 2
    W3 = W[None, ...] if orig_2d else W
    bsz, n_scales, n_t = W3.shape
    out_t = np.empty_like(W3)
    time_kernels = grilla["time_kernels"]
    for j in range(n_scales):
        k = time_kernels[j]
        half = (len(k) - 1) // 2
        for b in range(bsz):
            out_t[b, j] = np.convolve(W3[b, j], k, mode="full")[half:half + n_t]

    pad_s = grilla["n_sc_box"] // 2
    out_s = np.empty_like(out_t)
    for j in range(n_scales):
        j0, j1 = max(0, j - pad_s), min(n_scales, j + pad_s + 1)
        out_s[:, j, :] = out_t[:, j0:j1, :].mean(axis=1)
    return out_s[0] if orig_2d else out_s


def wavelet_coherence(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    grilla: dict | None = None,
    dt: float = 1.0 / 12.0,
    dj: float = 1.0 / 12.0,
    omega0: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Calcula coherencia wavelet al cuadrado R²(s, t) y diferencia de fase (en rad)."""
    x_arr = np.asarray(x_arr, dtype=float)
    y_arr = np.asarray(y_arr, dtype=float)
    n = x_arr.shape[-1]
    if y_arr.shape[-1] != n:
        raise ValueError(f"Longitudes incompatibles: {n} vs {y_arr.shape[-1]}")
    if grilla is None:
        grilla = grilla_morlet(n=n, dt=dt, dj=dj, omega0=omega0)

    Wx = cwt_morlet(x_arr, grilla)
    Wy = cwt_morlet(y_arr, grilla)
    inv_s = (1.0 / grilla["scales"])[..., None]
    Sxy = smooth_wt(inv_s * (Wx * np.conj(Wy)), grilla)
    Sxx = np.real(smooth_wt(inv_s * (np.abs(Wx) ** 2), grilla))
    Syy = np.real(smooth_wt(inv_s * (np.abs(Wy) ** 2), grilla))
    R2 = np.clip((np.abs(Sxy) ** 2) / np.maximum(Sxx * Syy, 1e-30), 0.0, 1.0)
    phase = np.angle(Sxy)
    return R2, phase, grilla


def simular_ar1(
    phi: float,
    b_size: int,
    n: int,
    rng: np.random.Generator,
    burn: int = 100,
) -> np.ndarray:
    """Simula b_size series AR(1) estandarizadas de largo n."""
    eps = rng.standard_normal((b_size, n + burn))
    out = np.empty_like(eps)
    out[:, 0] = eps[:, 0] / np.sqrt(max(1.0 - phi ** 2, 1e-6))
    for k in range(1, n + burn):
        out[:, k] = phi * out[:, k - 1] + eps[:, k]
    z_out = out[:, burn:]
    sd = np.maximum(z_out.std(axis=1, keepdims=True), 1e-12)
    return (z_out - z_out.mean(axis=1, keepdims=True)) / sd


def umbral_montecarlo_ar1(
    phi_x: float,
    phi_y: float,
    grilla: dict,
    n_mc: int | None = None,
    batch_mc: int = 25,
    burn: int = 100,
    borde: int = 60,
    seed: int = 0,
) -> tuple[np.ndarray, int]:
    """Estima el umbral al 95 % por escala mediante simulación Monte Carlo AR(1).

    Si `n_mc` es None, lee `ENSO_N_MC` del entorno (por defecto 250).
    """
    if n_mc is None:
        n_mc = int(os.environ.get("ENSO_N_MC", "250"))
    if n_mc < 1:
        raise ValueError(f"n_mc debe ser >= 1 (recibido {n_mc})")
    n = grilla["n"]
    rng = np.random.default_rng(seed)
    step = min(batch_mc, n_mc)
    r2_mc_list = []
    for b0 in range(0, n_mc, step):
        bs = min(step, n_mc - b0)
        x_mc = simular_ar1(phi_x, bs, n, rng, burn=burn)
        y_mc = simular_ar1(phi_y, bs, n, rng, burn=burn)
        r2_b, _, _ = wavelet_coherence(x_mc, y_mc, grilla=grilla)
        r2_mc_list.append(r2_b)
    r2_mc = np.concatenate(r2_mc_list, axis=0)
    sl = slice(borde, -borde) if 2 * borde < n else slice(None)
    sig95_scale = np.quantile(r2_mc[:, :, sl], 0.95, axis=(0, 2))
    return sig95_scale, n_mc


def fase_circular_media(phase: np.ndarray, mask: np.ndarray) -> dict:
    """Media circular de ángulos de fase en rad/grados, longitud resultante R y dispersión circular."""
    n_celdas = int(np.sum(mask))
    if n_celdas == 0:
        return {
            "n_celdas": 0,
            "fase_media_rad": float("nan"),
            "fase_media_grados": float("nan"),
            "longitud_resultante_R": float("nan"),
            "dispersion_circular": float("nan"),
        }
    vec = np.mean(np.exp(1j * phase[mask]))
    rad = float(np.angle(vec))
    r_res = float(np.abs(vec))
    return {
        "n_celdas": n_celdas,
        "fase_media_rad": rad,
        "fase_media_grados": float(np.degrees(rad)),
        "longitud_resultante_R": r_res,
        "dispersion_circular": float(1.0 - r_res),
    }
