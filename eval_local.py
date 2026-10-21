"""Puntúa el pipeline contra un ground truth propio usando la librería OFICIAL del concurso.

    python eval_local.py --official ../more-than-a-hack-2027 --data-dir synth/data --gt synth/data/gt.csv

Compara contra el baseline oficial (jugador más cercano + voto mayoritario de +/-50 frames).
Con el partido sintético sirve para verificar que el pipeline funciona; los números absolutos
no dicen nada sobre el partido real.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from config import PITCH, Params  # noqa: E402
from run import predict  # noqa: E402
from src import atd  # noqa: E402


def baseline_labels(data, window=50):
    """El baseline oficial reimplementado sobre los arrays ya cargados."""
    d = np.linalg.norm(data["players_xy"] - data["ball_xy"][:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    team = data["team_of_slot"][np.argmin(d, axis=1)]
    codes = np.where(np.isnan(data["ball_xy"][:, 0]), -1, team)
    h = np.concatenate(([0], np.cumsum(codes == 0)))
    a = np.concatenate(([0], np.cumsum(codes == 1)))
    i = np.arange(len(codes))
    lo, hi = np.maximum(0, i - window), np.minimum(len(codes), i + window + 1)
    hh, aa = h[hi] - h[lo], a[hi] - a[lo]
    return np.where(hh + aa == 0, "D", np.where(hh >= aa, "H", "A"))


def score(pm, gt_codes, labels):
    r = pm.evaluate(gt_codes, pm.label_codes(labels), bootstrap_n=0)
    return r


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", default=os.path.expanduser("~/mtah"),
                        help="clon del repo oficial more-than-a-hack-2027")
    parser.add_argument("--data-dir", default=os.path.join(HERE, "synth", "data"))
    parser.add_argument("--gt", default=os.path.join(HERE, "synth", "data", "gt.csv"))
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(args.official, "evaluation"))
    import possession_metrics as pm  # noqa: E402

    gt = pd.read_csv(args.gt, dtype={"label": "object"})
    gt_codes = pm.label_codes(gt)
    folder = os.path.join(args.data_dir, "tactical")
    data = atd.load(os.path.join(folder, "BAR-ATH_tactical_atd.xml"),
                    os.path.join(folder, "BAR-ATH_tactical_atd.txt"), len(gt), PITCH)

    ours, debug = predict(data, Params())
    rows = [("baseline_smooth50", baseline_labels(data)), ("pipeline", ours)]
    print("{:<20} {:>8} {:>8} {:>6} {:>7} {:>9} {:>8}".format(
        "", "spellF1", "macroF1", "poss", "dur(s)", "aliveF1", "teamAcc"))
    first = True
    for name, labels in rows:
        r = score(pm, gt_codes, labels)
        if first:
            print("{:<20} {:>8} {:>8} {:>6} {:>7.1f}".format(
                "ground truth", "", "", r["n_gt_spells"], r["gt_mean_spell_duration_s"]))
            first = False
        print("{:<20} {:>8.4f} {:>8.4f} {:>6} {:>7.1f} {:>9.4f} {:>8.4f}".format(
            name, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"],
            r["mean_spell_duration_s"], r["alive_dead_f1"], r["team_acc_gt_alive"]))
