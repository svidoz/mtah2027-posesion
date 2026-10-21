"""Pasos 5 y 6: posesión por equipo, adaptando la sección 2.3 del paper.

Paso 5, frames de control y ganancias
  - Zona de posesión (PZ): círculo de radio r_pz alrededor de cada jugador. Como el ATD no tiene
    identidades estables, trabajamos por EQUIPO: alcanza con saber qué equipo tiene al jugador
    más cercano dentro de la PZ.
  - Zona de duelo (DZ): si hay jugadores de los dos equipos a menos de r_dz, es duelo. Un frame de
    duelo nunca produce una ganancia por sí solo.
  - Ganancia (paper 2.3.3): un tramo de control de un equipo es ganancia solo si la pelota cambió
    de dirección o de velocidad durante el tramo, o si el control se sostuvo (conducción). Si la
    pelota pasa cerca sin alterarse, es un falso positivo y se descarta.

Paso 6, máquina de estados
  - La posesión se mantiene hasta una ganancia del rival. Las pelotas sueltas y los huecos de
    tracking quedan para el último dueño, que es lo que pide CHALLENGE.md.
  - Una posesión rival corta entre dos del mismo equipo se absorbe: es un toque sin control o un
    pase desviado, que según las reglas del concurso no cambia la posesión.
"""

import numpy as np

from config import frames
from src.ball import runs_of

HOME, AWAY, DUEL, NONE = 0, 1, 2, -1
LETTER = {HOME: "H", AWAY: "A"}


def control(players_xy, team_of_slot, ball, params):
    """Por frame: ctrl (HOME/AWAY/DUEL/NONE) y distancia del más cercano de cada equipo."""
    xy = ball["xy"].astype(np.float32)
    d = np.linalg.norm(players_xy - xy[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    home = d[:, team_of_slot == HOME].min(axis=1)
    away = d[:, team_of_slot == AWAY].min(axis=1)

    ctrl = np.full(len(xy), NONE, dtype=np.int8)
    near = np.minimum(home, away) < params.r_pz
    ctrl[near & (home <= away)] = HOME
    ctrl[near & (away < home)] = AWAY
    ctrl[(home < params.r_dz) & (away < params.r_dz)] = DUEL
    ctrl[~ball["known"]] = NONE
    return ctrl, home, away


def _unit(v):
    n = np.hypot(v[0], v[1])
    return None if not np.isfinite(n) or n < 1.0 else v / n   # < 1 m/s: dirección indefinida


def gains(ctrl, ball, params):
    """[(frame, team)] ordenadas: inicio de cada tramo de control que es ganancia real."""
    vel, speed = ball["vel"], ball["speed"]
    n = len(ctrl)
    dribble = frames(params.dribble_s)
    out = []
    for team in (HOME, AWAY):
        for f0, fn in runs_of(ctrl == team):
            if fn - f0 >= dribble:
                out.append((int(f0), team))
                continue
            lo, hi = max(0, f0 - 3), min(n, fn + 3)
            s = speed[lo:hi]
            s = s[np.isfinite(s)]
            speed_change = len(s) > 0 and (s.max() - s.min()) > params.gain_dv
            u_in, u_out = _unit(vel[max(0, f0 - 2)]), _unit(vel[min(n - 1, fn + 1)])
            turn = u_in is not None and u_out is not None and \
                float(np.dot(u_in, u_out)) < params.gain_cos_max
            if speed_change or turn:
                out.append((int(f0), team))
    out.sort()
    return out


def _spells(labels, start, stop):
    """[(s, e, letra)] tramos constantes dentro de [start, stop)."""
    seg = labels[start:stop]
    change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
    bounds = np.concatenate(([0], change, [len(seg)]))
    return [(start + a, start + b, seg[a]) for a, b in zip(bounds[:-1], bounds[1:])]


def absorb_short(labels, start, stop, min_len, end_len):
    """Una posesión corta entre dos del mismo equipo pasa a ese equipo. Una posesión corta justo
    antes de la pelota muerta (desvío que se va afuera) pasa al equipo anterior."""
    changed = True
    while changed:
        changed = False
        sp = _spells(labels, start, stop)
        for i in range(1, len(sp)):
            s, e, team = sp[i]
            prev = sp[i - 1][2]
            last = i == len(sp) - 1
            if (not last and e - s < min_len and sp[i + 1][2] == prev) or \
               (last and e - s < end_len):
                labels[s:e] = prev
                changed = True
                break


def assign(n, gain_list, dead, ball, ctrl, params):
    """Etiqueta final H/A/D por frame."""
    labels = np.full(n, "D", dtype="<U1")
    g_frames = np.array([g[0] for g in gain_list], dtype=np.int64)
    g_teams = np.array([g[1] for g in gain_list], dtype=np.int8)
    previous = HOME

    for start, stop in runs_of(~dead):
        idx = np.flatnonzero((g_frames >= start) & (g_frames < stop))
        if len(idx):
            current = int(g_teams[idx[0]])          # dueño de la reanudación
        else:                                       # sin ganancias: el que más controló
            seg = ctrl[start:stop]
            h, a = int(np.sum(seg == HOME)), int(np.sum(seg == AWAY))
            current = previous if h == a else (HOME if h > a else AWAY)

        labels[start:stop] = LETTER[current]
        for k in idx[1:]:
            team, f = int(g_teams[k]), int(g_frames[k])
            if team == current:
                continue
            if params.gap_switch == "mid":          # el cambio pasó en algún momento del hueco
                g = f
                while g > start and not ball["known"][g - 1]:
                    g -= 1
                f = (g + f) // 2
            labels[f:stop] = LETTER[team]
            current = team

        absorb_short(labels, start, stop, frames(params.min_spell_s), frames(0.8))
        previous = HOME if labels[stop - 1] == "H" else AWAY
    return labels
