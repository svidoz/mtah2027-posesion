"""Parámetros del pipeline, todos en un solo lugar.

Unidades: metros, segundos y m/s. Los umbrales en frames se derivan de FPS.
Los valores por defecto salen del paper (Vidal-Codina et al. 2022) agrandados para tracking
automático de video, como sugiere el propio paper. Hay que recalibrarlos con la data real:
NO los ajusten mirando solo el spellF1 de los 5 minutos públicos (ver EVALUATION.md).
"""

from dataclasses import dataclass

FPS = 25
PITCH = (105.3, 68.0)          # largo x ancho en metros (games.py del repo oficial)
N_FRAMES = 199319              # filas de la entrega
GRADED = [(11185, 90651), (110808, 190197)]   # frames puntuados (inclusive), DATA.md


@dataclass
class Params:
    # --- paso 3: pelota -------------------------------------------------------------------
    interp_max_gap_s: float = 0.4      # huecos de pelota más cortos que esto se interpolan
    savgol_window: int = 7             # ventana Savitzky-Golay (paper: 7 frames a 25 Hz)
    savgol_order: int = 2

    # --- paso 5: posesión (paper, sección 2.3) --------------------------------------------
    r_pz: float = 1.5                  # radio de zona de posesión (paper: 0.5-1 m en estadio)
    r_dz: float = 1.5                  # radio de zona de duelo
    gain_cos_max: float = 0.85         # cambio de dirección: cos(ángulo) < esto  (~32°)
    gain_dv: float = 2.0               # cambio de velocidad mínimo (m/s) dentro del control
    dribble_s: float = 0.6             # control sostenido este tiempo cuenta como ganancia

    # --- paso 6: máquina de estados -------------------------------------------------------
    min_spell_s: float = 1.2           # posesiones más cortas, entre dos del mismo equipo, se absorben
    gap_switch: str = "end"            # dónde poner el cambio si ocurrió durante un hueco de pelota

    # --- paso 4: pelota muerta ------------------------------------------------------------
    out_margin: float = 0.4            # metros más allá de la línea para declarar pelota afuera
    restart_speed: float = 4.0         # velocidad de pelota (m/s) que marca la reanudación
    restart_max_s: float = 60.0        # tope de una detención sin reanudación detectada
    throwin_margin: float = 0.3        # jugador más allá de la línea lateral = config. de lateral
    stop_speed: float = 1.2            # velocidad media de jugadores para considerar juego parado
    stop_min_s: float = 4.0            # duración mínima de juego parado sin pelota afuera


DEFAULT = Params()


def frames(seconds):
    return max(1, int(round(seconds * FPS)))
