# Posesión frame a frame — More than a Hack 2027

Pipeline para la Fase 1 del concurso: decir, para cada frame del video táctico, si la pelota es
del local (`H`), del visitante (`A`) o está muerta (`D`).

Adapta el paso de posesión del paper de Vidal-Codina et al. (2022), *Automatic event detection in
football using tracking data*, al tracking automático y ruidoso del concurso.

## Estructura

```
config.py              todos los parámetros, en metros y segundos
run.py                 ATD -> submission.csv
eval_local.py          puntúa con la librería oficial contra un ground truth propio
src/atd.py             lectura del ATD (paso 1)
src/ball.py            paso 3: interpolar huecos cortos, Savitzky-Golay, velocidad
src/deadball.py        paso 4: pelota afuera, configuración de lateral, juego parado, reanudación
src/possession.py      pasos 5 y 6: zona de posesión, ganancias, máquina de estados
synth/make_synthetic.py  partido sintético en el formato del ATD, con ground truth
```

## Uso

```
pip install numpy pandas scipy

# con la data real (carpeta data/ del repo oficial)
python run.py --data-dir ../more-than-a-hack-2027/data --out submission.csv
python ../more-than-a-hack-2027/evaluation/validate_submission.py submission.csv

# sin data: probar con el partido sintético
python synth/make_synthetic.py
python eval_local.py --official ../more-than-a-hack-2027
```

## Cómo decide

1. **Pelota.** Huecos de hasta 0,4 s se interpolan; los largos quedan vacíos para no inventar
   cambios de dirección. Se suaviza y se calcula la velocidad.
2. **Pelota muerta.** Se declara cuando la pelota pasa una línea, cuando un jugador está fuera de
   la línea lateral (lateral) o cuando todos los jugadores están casi quietos varios segundos.
   Termina con la primera patada desde pelota quieta o invisible.
3. **Control.** Un equipo controla si su jugador más cercano está a menos de `r_pz` de la pelota.
   Si hay jugadores de los dos equipos cerca, es duelo y no cuenta.
4. **Ganancia.** Un tramo de control cambia la posesión solo si la pelota cambió de dirección o
   de velocidad, o si el control se sostuvo (conducción). Es el criterio del paper y coincide con
   la regla del concurso de que un toque sin control no cambia la posesión.
5. **Máquina de estados.** La posesión se mantiene hasta una ganancia del rival. Una posesión
   rival corta entre dos del mismo equipo se absorbe (desvío o rebote).

## Verificar apenas llegue la data

- Que el origen de las coordenadas 0-1 esté en una esquina (lo asume `src/deadball.py`).
- Cómo vienen los grupos faltantes en el `.txt` (`NaN` solo o `NaN,NaN,NaN`; el parser acepta
  los dos) y si la pelota trae un valor de altura.
- Si el XML de metadata trae algún dato de pelota en juego que nos ahorre el paso de pelota muerta.
- La distancia típica pelota-jugador en los pases, para fijar `r_pz`.

## Resultado en el partido sintético (3 semillas)

| | spellF1 baseline | spellF1 pipeline |
|---|---|---|
| semilla 7 | 0.278 | 0.439 |
| semilla 11 | 0.397 | 0.526 |
| semilla 23 | 0.356 | 0.500 |

El sintético solo prueba que el pipeline funciona. Los parámetros se calibran con la data real,
y nunca eligiendo entre versiones por el spellF1 de los 5 minutos públicos (ver EVALUATION.md
del repo oficial).

## Reglas a tener presentes

- Las etiquetas tienen que salir del algoritmo; nada corregido a mano.
- `python run.py` tiene que regenerar exactamente el mismo CSV entregado.
- La data del partido no se sube nunca al repo (`.gitignore` ya excluye `data/` y `cache/`).
