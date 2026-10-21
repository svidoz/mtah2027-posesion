"""Paso 4: detectar pelota muerta (D).

El paper tenía un flag de pelota en juego que venía del proveedor. El ATD del concurso, según
DATA.md, no lo trae, así que lo inferimos con tres disparadores:

  1. Pelota afuera: la pelota (observada) pasa una línea por más de out_margin metros.
  2. Configuración de lateral (paper 2.3.4, IFAB Ley 15): un jugador está fuera de la línea
     lateral. Mientras alguien está afuera, el juego está detenido.
  3. Juego parado: la velocidad media de los jugadores cae debajo de stop_speed durante más de
     stop_min_s (faltas, lesiones, festejos), sin pelota en movimiento.

Una detención termina en la reanudación: la primera vez que la pelota, dentro del campo, sale
disparada (velocidad > restart_speed) desde cerca de un jugador después de haber estado quieta o
invisible. Esa patada es el saque, el lateral o el tiro libre.

SUPUESTO A VERIFICAR CON LA DATA REAL: que las coordenadas normalizadas 0-1 tengan el origen en
una esquina del campo, de modo que "afuera" sea x<0, x>largo, y<0 o y>ancho.
"""

import numpy as np

from config import frames


def rolling_mean(x, w):
    x = np.where(np.isfinite(x), x, np.nan)
    valid = np.isfinite(x)
    c = np.concatenate(([0.0], np.cumsum(np.where(valid, x, 0.0))))
    k = np.concatenate(([0], np.cumsum(valid)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - w // 2), np.minimum(len(x), i + w // 2 + 1)
    cnt = k[hi] - k[lo]
    return np.where(cnt > 0, (c[hi] - c[lo]) / np.maximum(cnt, 1), np.nan)


def detect(ball, players_xy, players_speed, near_dist, pitch, params):
    """Máscara booleana (F,) True donde la pelota está muerta, más las señales para depurar."""
    length, width = pitch
    x, y = ball["xy"][:, 0], ball["xy"][:, 1]
    known, speed = ball["known"], ball["speed"]
    m = params.out_margin

    out = known & ((x < -m) | (x > length + m) | (y < -m) | (y > width + m))
    inside = known & ~out

    py = players_xy[:, :, 1]
    with np.errstate(invalid="ignore"):
        throwin = np.any((py < -params.throwin_margin) |
                         (py > width + params.throwin_margin), axis=1)

    mean_speed = rolling_mean(np.nanmean(np.where(np.isfinite(players_speed), players_speed,
                                                  np.nan), axis=1), frames(2.0))
    slow = (mean_speed < params.stop_speed) & ~(known & (speed > params.restart_speed))
    stopped = np.zeros(len(x), dtype=bool)
    min_stop = frames(params.stop_min_s)
    from src.ball import runs_of
    for s, e in runs_of(slow):
        if e - s >= min_stop:
            stopped[s:e] = True

    trigger = out | throwin | stopped
    kick = inside & (speed > params.restart_speed) & (near_dist < 3.0)
    calm = ~known | (speed < 1.5)

    dead = np.zeros(len(x), dtype=bool)
    n, f = len(x), 0
    max_wait = frames(params.restart_max_s)
    cooldown_until = 0       # tras un saque, el que sacó sigue afuera unos frames: no es otra detención
    while f < n:
        if not (out[f] or (trigger[f] and f >= cooldown_until)):
            f += 1
            continue
        start = f
        g, seen_calm = f + 1, False
        while g < n and g - start < max_wait:
            seen_calm |= bool(calm[g - 1])
            if seen_calm and kick[g] and not out[g]:
                break
            g += 1
        dead[start:g] = True
        cooldown_until = g + frames(3.0)
        f = g + 1
    return dead, {"out": out, "throwin": throwin, "stopped": stopped, "kick": kick}
