"""Partido sintético en el MISMO formato que el ATD del concurso, con ground truth.

Sirve para probar el pipeline de punta a punta antes de tener la data real. NO sirve para
calibrar parámetros: la física es de juguete. Lo que sí reproduce son las trampas del problema:

  - pelota invisible ~40% del tiempo, más seguido cuando hay gente cerca (oclusiones)
  - pases desviados por un rival que NO cambian la posesión
  - pases interceptados que SÍ la cambian
  - pelotas sueltas tras una disputa: la posesión queda del último dueño hasta que alguien controla
  - pelota afuera, lateral con el jugador parado fuera de la línea, faltas con juego parado

    python synth/make_synthetic.py --out synth/data --minutes 12

Escribe synth/data/tactical/BAR-ATH_tactical_atd.{xml,txt} y synth/data/gt.csv
"""

import argparse
import os

import numpy as np

L, W, FPS = 105.3, 68.0, 25
HOME, AWAY = 0, 1


class Match:
    def __init__(self, n_frames, rng):
        self.n, self.rng = n_frames, rng
        self.team = np.array([HOME] * 11 + [AWAY] * 11)
        grid = [(5, 34)] + [(x, y) for x in (22, 38) for y in (10, 26, 42, 58)] + [(48, 26), (48, 42)]
        home = np.array(grid, dtype=float)
        away = np.column_stack([L - home[:, 0], W - home[:, 1]])
        self.base = np.vstack([home, away])
        self.pos = self.base.copy()
        self.jitter = np.zeros_like(self.base)
        self.P = np.zeros((n_frames, 22, 2))
        self.B = np.full((n_frames, 2), np.nan)
        self.gt = np.full(n_frames, "X", dtype="<U1")
        self.H = np.zeros(n_frames, dtype=bool)      # frames donde la pelota está fuera de cámara
        self.ball = np.array([L / 2, W / 2])
        self.owner, self.holder = HOME, 9
        self.mode, self.timer = "hold", 1.0
        self.roles = {}
        self.stats = {"pass": 0, "intercept": 0, "deflect": 0, "tackle": 0, "out": 0, "foul": 0}

    # -- jugadores ---------------------------------------------------------------------------
    def move(self, i, target, vmax):
        d = target - self.pos[i]
        dist = np.hypot(*d)
        step = vmax / FPS
        self.pos[i] = target if dist <= step else self.pos[i] + d / dist * step

    def attack_dir(self, team):
        return np.array([1.0, 0.0]) if team == HOME else np.array([-1.0, 0.0])

    def nearest(self, point, team=None, exclude=()):
        d = np.hypot(*(self.pos - point).T)
        if team is not None:
            d[self.team != team] = np.inf
        for e in exclude:
            d[e] = np.inf
        return int(np.argmin(d))

    def update_players(self, f):
        if f % 50 == 0:
            self.jitter = self.rng.normal(0, 4, self.base.shape)
        shift = 0.4 * (self.ball - np.array([L / 2, W / 2]))
        slow = self.mode == "dead"
        for i in range(22):
            if i in self.roles:
                target, v = self.roles[i]
                self.move(i, target, v)
            else:
                target = np.clip(self.base[i] + shift + self.jitter[i], 1, [L - 1, W - 1])
                self.move(i, target, 1.0 if slow else 2.5)

    # -- pelota ------------------------------------------------------------------------------
    def start_flight(self, frm, end, speed, receiver, deflector=None, out=False):
        self.mode = "flight"
        self.fl = {"from": frm.copy(), "end": end.copy(), "v": speed, "rec": receiver,
                   "defl": deflector, "deflected": False, "out": out}
        self.roles = {receiver: (end, 6.0)}
        if deflector is not None:
            self.roles[deflector] = ((frm + end) / 2, 7.0)

    def choose_action(self):
        r, h = self.rng.random(), self.holder
        mates = [i for i in np.flatnonzero(self.team == self.owner) if i != h]
        if r < 0.06:                                   # pase que se va afuera por el lateral
            self.stats["out"] += 1
            side = 0.0 if self.ball[1] < W / 2 else W
            end = np.array([np.clip(self.ball[0] + self.rng.normal(0, 10), 2, L - 2),
                            side + (-4 if side == 0 else 4)])
            self.start_flight(self.ball, end, 14, self.nearest(end, self.owner, (h,)), out=True)
        elif r < 0.09:                                 # falta: juego parado, saca el mismo equipo
            self.stats["foul"] += 1
            self.enter_dead(self.ball.copy(), self.owner, self.rng.uniform(6, 12), throwin=False)
        elif r < 0.15:                                 # disputa: la pelota queda suelta
            self.stats["tackle"] += 1
            ang = self.rng.uniform(0, 2 * np.pi)
            self.mode, self.loose_v, self.loose_t = "loose", np.array([np.cos(ang), np.sin(ang)]) * 7, 0
            chasers = [self.nearest(self.ball, HOME, (h,)), self.nearest(self.ball, AWAY, (h,))]
            self.roles = {c: (self.ball + self.loose_v * 0.4, 6.0) for c in chasers}
        else:                                          # pase
            self.stats["pass"] += 1
            d = np.hypot(*(self.pos[mates] - self.ball).T)
            rec = mates[int(self.rng.choice(np.argsort(d)[:5]))]
            end = np.clip(self.pos[rec] + self.rng.normal(0, 1.5, 2), 1, [L - 1, W - 1])
            speed = self.rng.uniform(10, 20)
            u = self.rng.random()
            if u < 0.10:                               # interceptado
                self.stats["intercept"] += 1
                mid = self.ball + (end - self.ball) * self.rng.uniform(0.4, 0.8)
                rec = self.nearest(mid, 1 - self.owner)
                self.start_flight(self.ball, mid, speed, rec)
            elif u < 0.22:                             # desviado, no cambia la posesión
                self.stats["deflect"] += 1
                self.start_flight(self.ball, end, speed, rec,
                                  deflector=self.nearest((self.ball + end) / 2, 1 - self.owner))
            else:
                self.start_flight(self.ball, end, speed, rec)

    def enter_dead(self, spot, restart_team, duration, throwin):
        self.mode, self.dead_t, self.dead_len = "dead", 0.0, duration
        self.restart_team, self.spot, self.hidden = restart_team, spot, throwin
        taker = self.nearest(spot, restart_team)
        stand = spot.copy()
        if throwin:
            stand[1] = -0.8 if spot[1] < W / 2 else W + 0.8
        self.taker, self.stand = taker, stand
        self.roles = {taker: (stand, 4.0)}

    def step_ball(self, f):
        if self.mode == "hold":
            self.ball = self.pos[self.holder] + 0.5 * self.attack_dir(self.owner)
            self.roles = {self.holder: (self.pos[self.holder] + self.attack_dir(self.owner) * 5, 3.5)}
            presser = self.nearest(self.ball, 1 - self.owner)
            self.roles[presser] = (self.ball, 4.0)
            self.timer -= 1 / FPS
            if self.timer <= 0:
                self.choose_action()
            return self.owner

        if self.mode == "flight":
            fl = self.fl
            d = fl["end"] - self.ball
            dist = np.hypot(*d)
            step = fl["v"] / FPS
            if fl["defl"] is not None and not fl["deflected"] and \
                    np.hypot(*(self.ball - (fl["from"] + fl["end"]) / 2)) < step:
                ang = np.radians(self.rng.choice([-1, 1]) * self.rng.uniform(15, 30))
                rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
                fl["end"] = self.ball + rot @ d * 0.9
                fl["v"] *= 0.6
                fl["deflected"] = True
                self.roles[fl["rec"]] = (fl["end"], 7.0)
                del self.roles[fl["defl"]]
            if fl["out"] and (self.ball[1] < 0 or self.ball[1] > W):
                spot = np.array([self.ball[0], 0.0 if self.ball[1] < 0 else W])
                self.enter_dead(spot, 1 - self.owner, self.rng.uniform(6, 11), throwin=True)
                self.ball = self.ball + d / dist * step
                return "D"
            if dist <= step:
                self.ball = fl["end"].copy()
                rec = fl["rec"]
                if fl["out"]:   # no llegó a salir: la recibe el compañero
                    pass
                self.holder, self.owner = rec, int(self.team[rec])
                self.pos[rec] = self.ball - 0.5 * self.attack_dir(self.owner)
                self.mode, self.timer = "hold", self.rng.uniform(0.4, 2.5)
                return self.owner
            self.ball = self.ball + d / dist * step
            return self.owner

        if self.mode == "loose":
            self.loose_t += 1 / FPS
            sp = np.hypot(*self.loose_v)
            if sp > 0:
                self.loose_v = self.loose_v / sp * max(0.0, sp - 5 / FPS)
            self.ball = np.clip(self.ball + self.loose_v / FPS, 0.5, [L - 0.5, W - 0.5])
            for c in self.roles:
                self.roles[c] = (self.ball, 6.0)
            d = np.hypot(*(self.pos - self.ball).T)
            if self.loose_t > 0.3 and d.min() < 0.8:
                self.holder = int(np.argmin(d))
                self.owner = int(self.team[self.holder])
                self.mode, self.timer = "hold", self.rng.uniform(0.6, 2.0)
            return self.owner

        # dead
        self.dead_t += 1 / FPS
        visible_from = self.dead_len - 1.5
        if self.dead_t >= visible_from:
            self.ball = self.pos[self.taker] + (0.0 if self.hidden else 0.4) * self.attack_dir(self.restart_team)
        elif self.hidden:
            self.ball = self.spot.copy()        # la pelota existe, pero el tracking no la ve
            self.H[f] = True
        if self.dead_t >= self.dead_len:
            self.owner, self.holder = self.restart_team, self.taker
            mates = [i for i in np.flatnonzero(self.team == self.owner) if i != self.taker]
            d = np.hypot(*(self.pos[mates] - self.pos[self.taker]).T)
            rec = mates[int(np.argmin(d))]
            start = self.ball.copy()
            if self.hidden:
                start[1] = np.clip(start[1], 0.3, W - 0.3)
            self.ball = start
            self.start_flight(start, np.clip(self.pos[rec], 1, [L - 1, W - 1]), 11, rec)
            return self.owner
        return "D"

    def run(self, graded):
        for f in range(self.n):
            self.update_players(f)
            label = self.step_ball(f)
            self.P[f] = self.pos
            self.B[f] = self.ball
            if graded[0] <= f <= graded[1]:
                self.gt[f] = label if label == "D" else ("H" if label == HOME else "A")


def observe(match, rng, miss_target=0.41):
    """Ruido y oclusiones: lo que vería el tracking automático."""
    P = match.P + rng.normal(0, 0.10, match.P.shape)
    P[rng.random(P.shape[:2]) < 0.03] = np.nan
    B = match.B + rng.normal(0, 0.12, match.B.shape)
    visible, hidden = True, np.zeros(match.n, dtype=bool)
    for f in range(match.n):
        crowd = np.sum(np.hypot(*(match.P[f] - match.B[f]).T) < 4) if np.isfinite(match.B[f, 0]) else 0
        if visible and rng.random() < 0.016 + 0.03 * (crowd >= 3):
            visible = False
        elif not visible and rng.random() < 0.03:
            visible = True
        hidden[f] = not visible
    B[hidden | match.H] = np.nan
    return P, B


def write_atd(folder, stem, P, B):
    os.makedirs(folder, exist_ok=True)
    teams = ('<Team id="T_H"><Name>Local sintético</Name></Team>'
             '<Team id="T_A"><Name>Visitante sintético</Name></Team>')
    players = "".join('<Player id="P{0}" teamId="{1}"><Name>Track {0}</Name><ShirtNumber>{0}</ShirtNumber>'
                      '</Player>'.format(i + 1, "T_H" if i < 11 else "T_A") for i in range(P.shape[1]))
    with open(os.path.join(folder, stem + "_atd.xml"), "w") as h:
        h.write('<?xml version="1.0"?><Main><Metadata><Teams>{}</Teams><Players>{}</Players>'
                '</Metadata></Main>'.format(teams, players))

    speed_p = np.vstack([np.zeros((1, P.shape[1])), np.hypot(*np.diff(P, axis=0).transpose(2, 0, 1)) * FPS])
    speed_b = np.concatenate([[0], np.hypot(*np.diff(B, axis=0).T) * FPS])
    with open(os.path.join(folder, stem + "_atd.txt"), "w") as h:
        for f in range(len(B)):
            groups = []
            for i in range(P.shape[1]):
                x, y = P[f, i]
                groups.append("NaN,NaN,NaN" if np.isnan(x) else
                              "{:.5f},{:.5f},{:.2f}".format(x / L, y / W, np.nan_to_num(speed_p[f, i])))
            bx, by = B[f]
            ball = "NaN" if np.isnan(bx) else "{:.5f},{:.5f},{:.2f}".format(
                bx / L, by / W, np.nan_to_num(speed_b[f]))
            h.write("{}:{}:{}\n".format(f + 1, ";".join(groups), ball))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data"))
    parser.add_argument("--minutes", type=float, default=12)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n = int(args.minutes * 60 * FPS)
    graded = (250, n - 250)
    m = Match(n, rng)
    m.run(graded)
    P, B = observe(m, rng)
    write_atd(os.path.join(args.out, "tactical"), "BAR-ATH_tactical", P, B)
    with open(os.path.join(args.out, "gt.csv"), "w") as h:
        h.write("frame,label\n")
        h.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(m.gt))

    g = m.gt[graded[0]:graded[1] + 1]
    change = np.flatnonzero(g[1:] != g[:-1]) + 1
    bounds = np.concatenate(([0], change, [len(g)]))
    poss = [(e - s) / FPS for s, e in zip(bounds[:-1], bounds[1:]) if g[s] in "HA"]
    print("frames {}  graded {}..{}".format(n, *graded))
    print("eventos", m.stats)
    print("posesiones {}  duración media {:.1f} s  D {:.0%}".format(len(poss), np.mean(poss), np.mean(g == "D")))
    print("pelota observada en {:.1%} de los frames graded".format(
        np.mean(~np.isnan(B[graded[0]:graded[1] + 1, 0]))))
