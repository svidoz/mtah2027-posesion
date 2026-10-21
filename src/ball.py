"""Paso 3: limpiar la trayectoria de la pelota.

1. Interpolar huecos cortos (la pelota se pierde un instante por oclusión).
2. Suavizar con Savitzky-Golay, como el paper (polinomio grado 2, ventana 7 frames).
3. Derivar velocidad y dirección por diferencias finitas, solo dentro de tramos continuos.

Los huecos largos se dejan en NaN a propósito: inventar la pelota durante 3 segundos crea
cambios de dirección falsos, y cada cambio falso es una posible ganancia de posesión falsa.
"""

import numpy as np
from scipy.signal import savgol_filter

from config import FPS, frames


def runs_of(mask):
    """[(start, stop)] de los tramos True de una máscara booleana (stop exclusivo)."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def interpolate_short_gaps(xy, max_gap):
    xy = xy.copy()
    seen = ~np.isnan(xy[:, 0])
    for start, stop in runs_of(~seen):
        if start == 0 or stop >= len(xy) or stop - start > max_gap:
            continue
        a, b = xy[start - 1], xy[stop]
        t = (np.arange(start, stop) - (start - 1)) / (stop - (start - 1))
        xy[start:stop] = a + (b - a) * t[:, None]
    return xy


def clean(ball_xy, params):
    """dict con xy (suavizada), vel (F,2) m/s, speed (F,), observed (F,) bool."""
    observed = ~np.isnan(ball_xy[:, 0])
    xy = interpolate_short_gaps(ball_xy.astype(np.float64), frames(params.interp_max_gap_s))

    known = ~np.isnan(xy[:, 0])
    vel = np.full_like(xy, np.nan)
    w = params.savgol_window
    for start, stop in runs_of(known):
        n = stop - start
        seg = xy[start:stop]
        if n >= w:
            xy[start:stop] = savgol_filter(seg, w, params.savgol_order, axis=0)
            vel[start:stop] = savgol_filter(seg, w, params.savgol_order, deriv=1,
                                            delta=1.0 / FPS, axis=0)
        elif n >= 2:
            vel[start:stop] = np.gradient(seg, 1.0 / FPS, axis=0)
    speed = np.hypot(vel[:, 0], vel[:, 1])
    return {"xy": xy, "vel": vel, "speed": speed, "observed": observed, "known": known}
