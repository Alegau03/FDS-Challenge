## Feature engineering (dettaglio)

Le feature si dividono in tre blocchi: statiche, dinamiche, e derivate/differenziali, cui si aggiungono type-chart e speed control.

### Statiche (team-level)

- Aggregati sulle statistiche base del team (per p1 e p2):
  - Medie, deviazioni standard e massimi per: base_hp, base_atk, base_def, base_spa, base_spd, base_spe.
  - Numero di tipi unici nel team (escludendo "notype").
  - Indici compositi:
    - Offense index: base_atk + base_spa (per Pokémon), poi media/std/max sul team.
    - Defense index: base_hp + base_def + base_spd (per Pokémon), poi media/std/max.
    - Speed index: base_spe media/std/max.
  - Rapporto offense/defense medio del team.
  - Entropia dei tipi del team (misura di varietà di typing).

### Dinamiche (timeline ≤ 30 turni)

- Finestre temporali: 1–10 (w1), 11–20 (w2), 21–30 (w3).
- Metriche per giocatore (p1 e p2), totali e per-finestra:
  - damage_dealt (danno inflitto; da delta `hp_pct` dell’avversario nel tempo).
  - fainted_pokemon (KO inflitti; dal raggiungimento di `hp_pct == 0`).
  - switches; forced_switches (euristico su danni subiti elevati nel turno precedente).
  - boosts totali e boost per singola stat: atk/def/spa/spd/spe (somma dei boost positivi).
  - status_inflicted (onset di status) e status_turns_inflicted (turni in cui l’avversario è sotto status).
  - priority_moves (mosse con priorità > 0).
  - lead_changes (cambi di leadership via danno cumulativo inflitto).
  - time_to_first_ko_inflicted (indice temporale del primo KO inflitto o -1 se assente).
- EMA del danno (ema_damage_dealt) con alpha=0.3 per catturare il momentum recente.
- Rate per turno: varianti normalizzate per numero di turni effettivi.

### Type-chart advantage

- Efficacia offensiva attesa dei lead (p1→p2 e p2→p1) basata su una matrice `TYPE_CHART` semplificata.
- Se i tipi del lead mancano, fallback ai tipi presenti nel team.
- Feature:
  - `p1_type_off_mult`, `p2_type_off_mult` (moltiplicatori ≥ 0).
  - `diff_type_off_mult` = p1 − p2.
- Nota: il type chart è semplificato; può essere esteso ai 18 tipi completi per maggiore fedeltà.

### Lead matchup & speed control

- Estrazione `base_spe` dei lead (se presenti):
  - `p1_lead_base_spe`, `p2_lead_base_spe`, `diff_lead_base_spe`.
  - `p1_outspeed_prob`: proxy di probabilità di outspeed via sigmoide della differenza di speed.
  - `diff_outspeed_prob` simmetrica.

### Feature differenziali

- Per ogni feature che inizia con `p1_` creiamo `diff_* = p1_* − p2_*` (statiche e dinamiche). Questo canale informativo è molto utile per GBDT.

### Dettagli implementativi e formule (allineati al codice)

- Finestre temporali: w1 = turni 1–10, w2 = 11–20, w3 = 21–30.
- Danno inflitto (per turno t):
  - $\text{damage}_{p1,t} = \max\{0, \mathrm{hp}^{\text{prev}}_{p2} - \mathrm{hp}_{p2,t}\}$ e simmetricamente per p2. Gli HP sono clampati in [0,100]. Le cure non contribuiscono (si usa max con 0).
- EMA del danno: $\mathrm{ema}_t = \alpha\ \text{damage}_t + (1-\alpha)\ \mathrm{ema}_{t-1}$ con $\alpha = 0.3$.
- KO inflitti: incremento quando $\mathrm{hp}_{opp} = 0$. Il tempo al primo KO è il primo turno con KO (altrimenti -1).
- Switch e forced switch: si incrementa `switches` quando cambia il nome del lead rispetto al turno precedente; `forced_switches` aumenta se nello stesso evento lo switch succede dopo che nel turno precedente il giocatore ha subito almeno 15 punti di danno.
- Status onset e persistenza:
  - `status_inflicted`: incremento quando lo status dell’avversario passa da `nostatus` a uno status attivo.
  - `status_turns_inflicted`: +1 in ogni turno in cui l’avversario è sotto uno status attivo.
- Boost e boost per-stat: somma dei boost positivi totali e per ciascuna stat in {atk, def, spa, spd, spe}.
- Lead changes: definiamo i danni cumulativi $D_{p1}, D_{p2}$ e un indicatore di leadership $L_t=\text{sign}(D_{p1}-D_{p2})$. `lead_changes` aumenta quando $L_t$ cambia segno rispetto a $L_{t-1}$.
- Rate per turno: per ogni metrica base m, $m\_\text{per\_turn} = m\_\text{totale} / N\_{turni}$.
- Type-chart advantage: dato l’insieme dei tipi attaccanti A e difendenti B,
  - $\text{mult}(A\to B) = \prod\limits_{a\in A}\prod\limits_{b\in B} \mathrm{TYPE\_CHART}[a][b]$ con default 1.0 se la coppia non è presente. `diff_type_off_mult = p1\_mult − p2\_mult`.
- Speed control e outspeed:
  - `diff_lead_base_spe = p1_lead_base_spe − p2_lead_base_spe`.
  - $p(\text{p1 outspeed}) = \sigma(k\cdot(\mathrm{spe1}-\mathrm{spe2}))$ con $k=0.05$; `diff_outspeed_prob = 2p − 1`.
- Nomenclatura colonne:
  - Suffix finestra: `_w1`, `_w2`, `_w3` per 1–10, 11–20, 21–30.
  - Variante normalizzata: `_per_turn`.
  - Dinamiche sono prefissate da `p1_`/`p2_` dopo la normalizzazione di prospettiva (vedi sotto), le differenziali da `diff_`.
- Prospettiva del player: se `player_is_p1=false` (o `player_side='p2'`) scambiamo p1/p2 per feature statiche e dinamiche PRIMA di generare i `diff_`, così `p1_*` rappresenta sempre il giocatore del target `player_won`.
## Catalogo delle feature (completo) 🧭

Questa sezione elenca tutte le feature generate, con pattern dei nomi e significato. Prefissi sempre coerenti: `p1_` e `p2_` indicano il lato del “player” normalizzato (vedi prospettiva). Per ciascuna `p1_*` esiste la corrispondente differenziale `diff_* = p1_* − p2_*`.

1) Statiche di team (per ciascun lato p1/p2)
- Base stats aggregate (per `stat ∈ {base_hp, base_atk, base_def, base_spa, base_spd, base_spe}`):
  - `{prefix}_team_mean_{stat}`
  - `{prefix}_team_std_{stat}`
  - `{prefix}_team_max_{stat}`
- Tipi e diversità:
  - `{prefix}_team_unique_types` (conteggio di tipi distinti, escluso `notype`)
  - `{prefix}_team_type_entropy` (entropia della distribuzione dei tipi)
- Indici compositi:
  - `{prefix}_team_mean_offense`, `{prefix}_team_std_offense`, `{prefix}_team_max_offense` con offense = base_atk + base_spa
  - `{prefix}_team_mean_defense`, `{prefix}_team_std_defense`, `{prefix}_team_max_defense` con defense = base_hp + base_def + base_spd
  - `{prefix}_team_mean_speed`, `{prefix}_team_std_speed`, `{prefix}_team_max_speed` con base_spe
  - `{prefix}_team_offense_defense_ratio` = mean_offense / mean_defense

2) Dinamiche su timeline (≤ 30 turni) – totali, per finestra e rate per turno
- Metriche base M = {damage_dealt, boosts, fainted_pokemon, switches, status_inflicted, priority_moves}
  - Totali: `{prefix}_{M}`
  - Per finestra: `{prefix}_{M}_w1`, `{prefix}_{M}_w2`, `{prefix}_{M}_w3` (turni 1–10, 11–20, 21–30)
  - Rate per turno: `{prefix}_{M}_per_turn` = `{prefix}_{M}` / N_turni
- Altre dinamiche:
  - `{prefix}_status_turns_inflicted` (turni in cui l’avversario è sotto status)
  - `{prefix}_ema_damage_dealt` (EMA del danno con α=0.3)
  - `{prefix}_boosts_atk_sum`, `{prefix}_boosts_def_sum`, `{prefix}_boosts_spa_sum`, `{prefix}_boosts_spd_sum`, `{prefix}_boosts_spe_sum` (somme dei boost positivi per stat)
  - `{prefix}_lead_changes` (variazioni di leadership basate sul danno cumulativo)
  - `{prefix}_time_to_first_ko_inflicted` (turno del primo KO inflitto, -1 se nessuno)
  - `{prefix}_forced_switches` (switch stimati come forzati dopo danno subito ≥ 15 nel turno precedente)

3) Type-chart advantage (lead matchup)
- `p1_type_off_mult`, `p2_type_off_mult` (moltiplicatori offensivi attesi dei lead; fallback ai tipi medi di team)
- `diff_type_off_mult` = p1 − p2

4) Speed control del lead
- `p1_lead_base_spe`, `p2_lead_base_spe`, `diff_lead_base_spe`
- `p1_outspeed_prob` = σ(0.05 · (spe1 − spe2))
- `diff_outspeed_prob` = 2·`p1_outspeed_prob` − 1

5) Feature differenziali (automatiche)
- Per ogni `p1_*` (statiche o dinamiche) generiamo `diff_* = p1_* − p2_*`.

Note:
- Se la timeline manca, le dinamiche sono 0 e restano solo le statiche.
- I percorsi di naming sono coerenti tra train e test grazie alla normalizzazione della prospettiva.
