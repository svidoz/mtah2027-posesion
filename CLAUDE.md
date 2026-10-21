# Contexto del proyecto

Equipo argentino participando en **More than a Hack 2027 — Fase 1: Ball Possession**
(Metrica Sports, FC Barcelona, Mobile World Capital Barcelona). Respondé en español rioplatense.

- Repo oficial: https://github.com/metrica-sports/more-than-a-hack-2027 — clonarlo al lado de este
  (`../more-than-a-hack-2027`). Leer CHALLENGE.md, DATA.md, EVALUATION.md, RULES.md.
- Entrega: 17 de noviembre de 2026. CSV `frame,label` de 199.319 filas + este repo público + documento técnico.
- La data va en `../more-than-a-hack-2027/data/` (gated, licenciada). **Nunca commitearla ni publicar
  derivados frame a frame** (tracking exportado, clips). `.gitignore` excluye `data/`, `cache/`, `*.csv`.

## La tarea

Para cada frame del video táctico (25 fps): `H` local (Barcelona) con pelota viva, `A` visitante
(Athletic) con pelota viva, `D` pelota muerta, `X` sin respuesta. Partido: BAR-ATH, LaLiga 2024-25.
Frames puntuados: 11185–90651 y 110808–190197. Ground truth público: solo 11185–18684 (17 posesiones,
media 14,2 s) en `evaluation/gt/` del repo oficial.

Reglas de posesión del concurso: con pelota viva siempre hay un dueño; en pelota suelta lo conserva el
último equipo; cambia solo cuando el rival la tiene **bajo control**. No cambian la posesión: un toque sin
control, un pase desviado no interceptado, un duelo aéreo ganado sin control posterior. El ground truth
sale del tracking corregido, no de video ni de eventos.

## La métrica (clave para todas las decisiones)

- Ranking por **spellF1**: cada posesión es un objeto; emparejamiento 1 a 1 por IoU temporal, F1
  promediado en IoU 0,3–0,7. Fragmentar posesiones se castiga muchísimo. Desempate: macroF1 por frame.
- `X` en spellF1 se rellena con la última etiqueta; en macroF1 cuenta como error → en frames
  puntuados conviene mantener el último dueño en vez de poner `X`.
- Piso a superar: baseline_smooth50, spellF1 **0,2433** en el partido completo.
- **No elegir entre versiones por el spellF1 de los 5 minutos públicos**: con 17 posesiones es ruido
  (el propio repo muestra que elige mal la ventana del baseline). Usarlos solo para chequear que el
  pipeline está bien (alineación, formato, cantidad y duración de posesiones). Diferencias < 0,04 son ruido.

## La data (DATA.md)

- ATD raw: `<frame_id>:<x,y,s>;...;<x,y,s>:<x,y,s>` (jugadores; pelota). Coordenadas 0–1, velocidad m/s,
  NaN si falta. `video_frame = frame_id - 1`. Campo 105,3 × 68,0 m.
- 46 track slots sin identidad real (se cortan y cambian de identidad) → todo a nivel equipo.
- Pelota visible solo en 59,3 % de los frames puntuados. Sin eje z.
- Smart Tagging (SportsCode XML, segundos × 25 = frame): fases y pelotas paradas automáticas, con
  errores; los laterales tienen Team = N/A.
- Homografía por frame (pitch 0–1 → píxeles), válida 100 % en el táctico.
- Prohibido etiquetar a mano. `run.py` tiene que regenerar exactamente el CSV entregado.

## Base técnica

Paper: Vidal-Codina et al. 2022, *Automatic event detection in football using tracking data*
(arXiv 2202.00804). Del paso de posesión tomamos: zona de posesión alrededor de cada jugador, zona de
duelo, y **ganancia solo si la pelota cambia de dirección o velocidad dentro de la zona** (o control
sostenido). Suavizado Savitzky-Golay (orden 2, ventana 7). Disparadores de pelota parada por posiciones
de jugadores (lateral = jugador fuera de la línea, etc.). El paper usaba tracking de estadio limpio con
flag de pelota en juego; acá hay que inferirlo.

## Estado del plan

1. Entorno y data — **pendiente: la data todavía no llegó** (inscripción hecha / por confirmar).
2. Exploración de la data real — pendiente.
3. Limpieza de pelota — `src/ball.py` (hecho, sin calibrar).
4. Pelota muerta — `src/deadball.py` (hecho, sin calibrar).
5. Control y ganancias — `src/possession.py` (hecho, sin calibrar).
6. Máquina de estados — `src/possession.py` (hecho, sin calibrar).
7. Validación sin sobreajuste — pendiente.
8. Entrega (CSV validado, repo, documento técnico) — pendiente.

En el partido sintético (`synth/make_synthetic.py`, 3 semillas) el pipeline saca spellF1 0,44–0,53
contra 0,28–0,40 del baseline. El sintético solo prueba que funciona; no calibrar con él.

## Primero, apenas llegue la data

- Verificar que el origen de las coordenadas 0–1 esté en una esquina (lo asume `src/deadball.py`).
- Ver cómo vienen los grupos faltantes en el `.txt` y si la pelota trae altura.
- Ver si el XML trae algún indicador de pelota en juego.
- Medir la distancia típica pelota–jugador en recepciones para fijar `r_pz`.
- Correr el baseline oficial y nuestro `run.py`, y validar con `evaluation/validate_submission.py`.

## Ideas abiertas

- Usar Smart Tagging `SET PIECES` como evidencia de pelota muerta y de quién reanuda.
- En huecos largos de pelota, inferir su posición por convergencia de jugadores o desde el video.
- HMM / Viterbi sobre (H, A, D) con costo por transición, como alternativa a la máquina de estados.
