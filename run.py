"""Pipeline completo: ATD -> CSV de entrega (frame,label).

    python run.py --data-dir ../more-than-a-hack-2027/data --out submission.csv

Pasos (ver README.md):
  3. limpiar la pelota               src/ball.py
  4. detectar pelota muerta          src/deadball.py
  5. control y ganancias (paper)     src/possession.py
  6. máquina de estados              src/possession.py
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from config import DEFAULT, FPS, GRADED, N_FRAMES, PITCH, Params  # noqa: E402
from src import atd, ball as ball_mod, deadball, possession  # noqa: E402


def predict(data, params=DEFAULT, pitch=PITCH):
    n = len(data["ball_xy"])
    ball = ball_mod.clean(data["ball_xy"], params)
    ctrl, home_d, away_d = possession.control(data["players_xy"], data["team_of_slot"], ball, params)
    dead, signals = deadball.detect(ball, data["players_xy"], data["players_speed"],
                                    np.minimum(home_d, away_d), pitch, params)
    gain_list = possession.gains(ctrl, ball, params)
    labels = possession.assign(n, gain_list, dead, ball, ctrl, params)
    debug = {"ball": ball, "ctrl": ctrl, "dead": dead, "gains": gain_list, **signals}
    return labels, debug


def write_submission(path, labels):
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))


def summary(labels, graded=GRADED):
    """Lo mismo que conviene mirar antes de cada entrega: cuántas posesiones y cuánto duran."""
    n_poss, durations = 0, []
    graded = [(a, min(b, len(labels) - 1)) for a, b in graded if a < len(labels)] or \
        [(0, len(labels) - 1)]
    for a, b in graded:
        seg = labels[a:b + 1]
        change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        bounds = np.concatenate(([0], change, [len(seg)]))
        for s, e in zip(bounds[:-1], bounds[1:]):
            if seg[s] in "HA":
                n_poss += 1
                durations.append((e - s) / FPS)
    graded_labels = np.concatenate([labels[a:b + 1] for a, b in graded])
    share = {k: float(np.mean(graded_labels == k)) for k in "HAD"}
    return n_poss, (float(np.mean(durations)) if durations else 0.0), share


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Posesión frame a frame desde el ATD")
    parser.add_argument("--data-dir", required=True, help="carpeta data/ del concurso")
    parser.add_argument("--stem", default="BAR-ATH_tactical")
    parser.add_argument("--n-frames", type=int, default=N_FRAMES)
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--cache", default=os.path.join(HERE, "cache", "atd.npz"))
    args = parser.parse_args()

    folder = os.path.join(args.data_dir, "tactical")
    data = atd.load(os.path.join(folder, args.stem + "_atd.xml"),
                    os.path.join(folder, args.stem + "_atd.txt"),
                    args.n_frames, PITCH, cache=args.cache)
    print("equipos: H = {}, A = {}".format(*data["names"]))
    print("pelota observada en {:.1f}% de los frames".format(
        100 * np.mean(~np.isnan(data["ball_xy"][:, 0]))))

    labels, debug = predict(data, Params())
    write_submission(args.out, labels)
    n_poss, mean_dur, share = summary(labels)
    print("posesiones {}  duración media {:.1f} s  (verdad pública: 14.2 s)".format(n_poss, mean_dur))
    print("proporción H {H:.2f}  A {A:.2f}  D {D:.2f}".format(**share))
    print("escrito", args.out)
