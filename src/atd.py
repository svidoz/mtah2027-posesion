"""Lectura del Automatic Tracking Data (ATD) de Metrica: metadata XML + archivo raw.

Formato raw (según starter/baseline.py del repo oficial), una línea por frame de video:

    <frame_id>:<x,y,s>;<x,y,s>;...;<x,y,s>:<x,y,s>
               ^ un grupo por track slot      ^ pelota

Coordenadas normalizadas 0-1, velocidad en m/s, NaN para lo que no se ve. frame_id arranca en 1;
la fila de la entrega es frame_id - 1.

El parser no asume que un grupo faltante sea "NaN,NaN,NaN": también acepta "NaN" solo, y una
pelota con un cuarto valor (z) si llegara a venir. El resultado se cachea en .npz porque leer
199.319 líneas tarda un par de minutos.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

HOME, AWAY = 0, 1


def load_meta(meta_path):
    """(team_of_slot, (nombre_local, nombre_visitante)). El primer Team del XML es el local."""
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("se esperaban 2 equipos en {}, hay {}".format(meta_path, len(teams)))
    home_id = teams[0].get("id")
    names = tuple((t.findtext("Name") or t.get("id")) for t in teams)
    slots = [p.get("teamId") for p in root.iter("Player")]
    team_of_slot = np.array([HOME if t == home_id else AWAY for t in slots], dtype=np.int8)
    return team_of_slot, names


def _group(text, width):
    values = text.split(",")
    out = [float(v) for v in values[:width]]
    if len(out) < width:
        out += [np.nan] * (width - len(out))
    return out


def load_raw(raw_path, n_slots, n_frames):
    """players (F, S, 3) [x, y, speed] y ball (F, 3) [x, y, speed], normalizados 0-1, NaN si falta."""
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            row = int(parts[0]) - 1
            if not 0 <= row < n_frames:
                continue
            groups = parts[1].split(";")
            for slot, group in enumerate(groups[:n_slots]):
                if group and group[0] != "N":
                    players[row, slot] = _group(group, 3)
            if parts[2] and parts[2][0] != "N":
                ball[row] = _group(parts[2], 3)
    return players, ball


def load(meta_path, raw_path, n_frames, pitch, cache=None):
    """Todo en metros. Devuelve dict con players_xy (F,S,2), players_speed (F,S), ball_xy (F,2),
    ball_speed_raw (F,), team_of_slot (S,), names."""
    if cache and os.path.exists(cache):
        data = np.load(cache, allow_pickle=True)
        return {k: data[k] for k in data.files}

    team_of_slot, names = load_meta(meta_path)
    players, ball = load_raw(raw_path, len(team_of_slot), n_frames)
    scale = np.array(pitch, dtype=np.float32)
    out = {
        "players_xy": players[:, :, :2] * scale,
        "players_speed": players[:, :, 2],
        "ball_xy": ball[:, :2] * scale,
        "ball_speed_raw": ball[:, 2],
        "team_of_slot": team_of_slot,
        "names": np.array(names, dtype=object),
    }
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        np.savez_compressed(cache, **out)
    return out
