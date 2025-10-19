"""
Feature engineering 

Obiettivo pratico (in parole semplici): trasformare ogni battaglia JSON in una riga di
tabella con tante colonne (feature) utili ai modelli. Le feature includono:
- informazioni statiche sui team (medie/max delle statistiche base, diversità/entropia tipi, indici offense/defense/speed);
- informazioni dinamiche dai primi `max_turns` turni (default 30) con finestre [1–10], [11–20], [21–30],
  rate per turno, mosse prioritarie, switch (anche forzati), KO, boost, danni (anche EMA), status inflitti e loro persistenza;
- differenze p1 − p2 per molte metriche (prefisso diff_...);
- un vantaggio tipo vs tipo semplificato per i lead (type-chart) e una proxy di "outspeed" basata su base_spe.

Nota di prospettiva: p1_* rappresenta sempre il player (giocatore per cui si calcola player_won),
quindi se nel JSON il player è su p2, scambiamo p1/p2 per allineare le feature.

Assunzioni e fallback robusti:
- in test p2 può avere solo il lead: usiamo il lead come team minimo;
- il battle log può chiamarsi battle_log o battle_timeline;
- se mancano i tipi o sono 'notype', li ignoriamo nel conteggio dell'entropia;
- per l'outspeed usiamo una sigmoide sulla differenza di base_spe (proxy semplice, ma utile).
"""

import pandas as pd
import numpy as np
import json
from tqdm import tqdm

# --- Type chart  efficacia attacco -> moltiplicatore ---

TYPE_CHART = {
    'fire':    {'grass': 2.0, 'ice': 2.0, 'bug': 2.0, 'steel': 2.0, 'water': 0.5, 'rock': 0.5, 'dragon': 0.5, 'fire': 0.5},
    'water':   {'fire': 2.0, 'ground': 2.0, 'rock': 2.0, 'water': 0.5, 'grass': 0.5, 'dragon': 0.5},
    'grass':   {'water': 2.0, 'ground': 2.0, 'rock': 2.0, 'fire': 0.5, 'grass': 0.5, 'poison': 0.5, 'flying': 0.5, 'bug': 0.5, 'dragon': 0.5, 'steel': 0.5},
    'electric':{'water': 2.0, 'flying': 2.0, 'grass': 0.5, 'electric': 0.5, 'dragon': 0.5, 'ground': 0.0},
    'ice':     {'grass': 2.0, 'ground': 2.0, 'flying': 2.0, 'dragon': 2.0, 'fire': 0.5, 'water': 0.5, 'ice': 0.5, 'steel': 0.5},
    'fighting':{'normal': 2.0, 'ice': 2.0, 'rock': 2.0, 'dark': 2.0, 'steel': 2.0, 'poison': 0.5, 'flying': 0.5, 'psychic': 0.5, 'bug': 0.5, 'fairy': 0.5, 'ghost': 0.0},
    'ground':  {'fire': 2.0, 'electric': 2.0, 'poison': 2.0, 'rock': 2.0, 'steel': 2.0, 'grass': 0.5, 'bug': 0.5, 'flying': 0.0},
    'flying':  {'grass': 2.0, 'fighting': 2.0, 'bug': 2.0, 'electric': 0.5, 'rock': 0.5, 'steel': 0.5},
    'psychic': {'fighting': 2.0, 'poison': 2.0, 'psychic': 0.5, 'steel': 0.5, 'dark': 0.0},
    'bug':     {'grass': 2.0, 'psychic': 2.0, 'dark': 2.0, 'fire': 0.5, 'fighting': 0.5, 'poison': 0.5, 'flying': 0.5, 'ghost': 0.5, 'steel': 0.5, 'fairy': 0.5},
    'rock':    {'fire': 2.0, 'ice': 2.0, 'flying': 2.0, 'bug': 2.0, 'fighting': 0.5, 'ground': 0.5, 'steel': 0.5},
    'ghost':   {'psychic': 2.0, 'ghost': 2.0, 'dark': 0.5, 'normal': 0.0},
    'dragon':  {'dragon': 2.0, 'steel': 0.5, 'fairy': 0.0},
    'dark':    {'psychic': 2.0, 'ghost': 2.0, 'fighting': 0.5, 'dark': 0.5, 'fairy': 0.5},
    'steel':   {'ice': 2.0, 'rock': 2.0, 'fairy': 2.0, 'fire': 0.5, 'water': 0.5, 'electric': 0.5, 'steel': 0.5},
    'fairy':   {'fighting': 2.0, 'dragon': 2.0, 'dark': 2.0, 'fire': 0.5, 'poison': 0.5, 'steel': 0.5},
    'normal':  {'rock': 0.5, 'ghost': 0.0, 'steel': 0.5},
    'poison':  {'grass': 2.0, 'fairy': 2.0, 'poison': 0.5, 'ground': 0.5, 'rock': 0.5, 'ghost': 0.5, 'steel': 0.0},

}

def type_multiplier(attacking_types, defending_types):
    """Calcola un moltiplicatore di efficacia semplificato dato attaccante vs difensore.

    Parametri:
    - attacking_types: lista dei tipi dell'attaccante (es. ['fire', 'flying'])
    - defending_types: lista dei tipi del difensore (es. ['grass', 'steel'])

    Ritorna:
    - float: prodotto dei moltiplicatori dal TYPE_CHART (default 1.0 per coppie sconosciute)
    """
    mult = 1.0
    for atk in attacking_types or []:
        atk = (atk or '').lower()
        row = TYPE_CHART.get(atk, {})
        for df in defending_types or []:
            df = (df or '').lower()
            mult *= float(row.get(df, 1.0))
    return mult

def get_static_features(team_details, prefix):
    """Estrae feature statiche dal team.

    Spiegato semplice: guardiamo chi c'è nel team (anche solo il lead se siamo in test) e
    calcoliamo medie, massimi e dispersioni delle statistiche base, quanta varietà di tipi c'è,
    e tre indici intuitivi:
    - offense = base_atk + base_spa (quanto possiamo fare male)
    - defense = base_hp + base_def + base_spd (quanto reggiamo)
    - speed   = base_spe

    Parametri:
    - team_details: lista di dict con chiavi tipo base_hp/base_atk/... e 'types'
    - prefix: 'p1' o 'p2' per nominare le colonne

    Ritorna: dict di feature, es. {'p1_team_mean_base_hp': ..., 'p1_team_unique_types': ...}
    """
    features = {}
    stats = ['base_hp', 'base_atk', 'base_def', 'base_spa', 'base_spd', 'base_spe']
    
    all_stats_values = {stat: [] for stat in stats}
    all_types = []
    offense_vals = []
    defense_vals = []
    speed_vals = []

    if team_details:
        for pokemon in team_details:
            for stat in stats:
                all_stats_values[stat].append(pokemon.get(stat, 0))
            all_types.extend(pokemon.get('types', []))
            # Indici compositi
            atk = pokemon.get('base_atk', 0)
            spa = pokemon.get('base_spa', 0)
            hp = pokemon.get('base_hp', 0)
            de = pokemon.get('base_def', 0)
            sd = pokemon.get('base_spd', 0)
            spe = pokemon.get('base_spe', 0)
            offense_vals.append(atk + spa)
            defense_vals.append(hp + de + sd)
            speed_vals.append(spe)

    for stat, values in all_stats_values.items():
        if values:
            features[f'{prefix}_team_mean_{stat}'] = np.mean(values)
            features[f'{prefix}_team_std_{stat}'] = np.std(values)
            features[f'{prefix}_team_max_{stat}'] = np.max(values)
        else:
            features[f'{prefix}_team_mean_{stat}'] = 0
            features[f'{prefix}_team_std_{stat}'] = 0
            features[f'{prefix}_team_max_{stat}'] = 0

    # Diversità tipi nel team (ignoriamo 'notype')
    features[f'{prefix}_team_unique_types'] = len(set(t for t in all_types if t != 'notype'))

    # Indici compositi aggregati
    if offense_vals:
        features[f'{prefix}_team_mean_offense'] = float(np.mean(offense_vals))
        features[f'{prefix}_team_std_offense'] = float(np.std(offense_vals))
        features[f'{prefix}_team_max_offense'] = float(np.max(offense_vals))
    else:
        features[f'{prefix}_team_mean_offense'] = 0.0
        features[f'{prefix}_team_std_offense'] = 0.0
        features[f'{prefix}_team_max_offense'] = 0.0
    if defense_vals:
        features[f'{prefix}_team_mean_defense'] = float(np.mean(defense_vals))
        features[f'{prefix}_team_std_defense'] = float(np.std(defense_vals))
        features[f'{prefix}_team_max_defense'] = float(np.max(defense_vals))
    else:
        features[f'{prefix}_team_mean_defense'] = 0.0
        features[f'{prefix}_team_std_defense'] = 0.0
        features[f'{prefix}_team_max_defense'] = 0.0
    if speed_vals:
        features[f'{prefix}_team_mean_speed'] = float(np.mean(speed_vals))
        features[f'{prefix}_team_std_speed'] = float(np.std(speed_vals))
        features[f'{prefix}_team_max_speed'] = float(np.max(speed_vals))
    else:
        features[f'{prefix}_team_mean_speed'] = 0.0
        features[f'{prefix}_team_std_speed'] = 0.0
        features[f'{prefix}_team_max_speed'] = 0.0

    # Rapporto offense/defense: quanto "picchiamo" in media rispetto a quanto reggiamo
    denom = features[f'{prefix}_team_mean_defense'] if features[f'{prefix}_team_mean_defense'] != 0 else 1.0
    features[f'{prefix}_team_offense_defense_ratio'] = features[f'{prefix}_team_mean_offense'] / denom

    # Entropia dei tipi: più è alta, più il team è vario nei tipi (utile per coperture)
    types = [t for t in all_types if t != 'notype']
    if types:
        values, counts = np.unique(types, return_counts=True)
        probs = counts / counts.sum()
        type_entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    else:
        type_entropy = 0.0
    features[f'{prefix}_team_type_entropy'] = type_entropy
    return features

def get_dynamic_features(battle_log, max_turns=30):
    """Estrae feature dinamiche dai primi turni della battaglia (per p1 e p2).

    In breve, per i primi `max_turns` turni costruiamo:
    - finestre temporali w1 (1–10), w2 (11–20), w3 (21–30) per separare early/mid/late game;
    - conteggi di: switch, mosse prioritarie, status inflitti, KO, boost totali e per-stat;
    - danno inflitto e una EMA del danno (più peso agli ultimi turni);
    - stime di switch forzati (se nel turno prima abbiamo subito molto danno);
    - lead changes basati sul danno cumulativo (chi è “avanti” cambia durante il match?);
    - rate per turno per le metriche principali.

    Parametri:
    - battle_log: lista di turni (dict) con stati/mosse p1/p2; può essere anche "battle_timeline"
    - max_turns: quanti turni guardiamo al massimo (default 30)

    Ritorna:
    - (p1_feats: dict, p2_feats: dict)
    """
    # Inizializza le feature base
    base_metrics = ['damage_dealt', 'boosts', 'fainted_pokemon', 'switches', 'status_inflicted', 'priority_moves']
    def init_feat_dict():
        d = {m: 0 for m in base_metrics}
        # finestre
        for m in base_metrics:
            d[f'{m}_w1'] = 0
            d[f'{m}_w2'] = 0
            d[f'{m}_w3'] = 0
        # rate per turno (calcolati a fine pass)
        # status persistence
        d['status_turns_inflicted'] = 0
        # EMA del danno
        d['ema_damage_dealt'] = 0.0
        # boost per tipo (somme positive)
        for stat in ['atk', 'def', 'spa', 'spd', 'spe']:
            d[f'boosts_{stat}_sum'] = 0
        # dinamiche aggiuntive
        d['lead_changes'] = 0
        d['time_to_first_ko_inflicted'] = -1  # -1: nessun KO inflitto entro max_turns
        d['forced_switches'] = 0
        return d

    p1_feats = init_feat_dict()
    p2_feats = init_feat_dict()

    p1_last_hp = {}
    p2_last_hp = {}

    # Variabili per tracciare lo stato precedente
    p1_last_pkmn = None
    p2_last_pkmn = None
    p1_opponent_last_status = {}
    p2_opponent_last_status = {}

    if not battle_log:
        return p1_feats, p2_feats

    # Parametro per EMA (più alto = più peso agli ultimi turni)
    alpha = 0.3  # recency weighting

    # Limita ai primi `max_turns` turni
    n_turns = min(max_turns, len(battle_log))
    # Variabili per lead (vantaggio) e switch forzati
    p1_cum_damage = 0.0
    p2_cum_damage = 0.0
    last_leader = 0  # 1 se p1 in vantaggio, -1 se p2, 0 pari
    prev_turn_damage_taken_p1 = 0.0
    prev_turn_damage_taken_p2 = 0.0
    for t_idx, turn in enumerate(battle_log[:max_turns]):
        # Identifica finestra
        if t_idx < 10:
            w = 'w1'
        elif t_idx < 20:
            w = 'w2'
        else:
            w = 'w3'

        p1_state = turn.get('p1_pokemon_state', {})
        p2_state = turn.get('p2_pokemon_state', {})
        p1_move = turn.get('p1_move_details', {})
        p2_move = turn.get('p2_move_details', {})

        if p1_state and p2_state:
            p1_pkmn_name = p1_state.get('name')
            p2_pkmn_name = p2_state.get('name')

            # --- SWITCH ---
            if p1_last_pkmn and p1_pkmn_name != p1_last_pkmn:
                p1_feats['switches'] += 1
                p1_feats[f'switches_{w}'] += 1
                # euristica: switch forzato se nel turno precedente ha subito molto danno
                if prev_turn_damage_taken_p1 >= 15:
                    p1_feats['forced_switches'] += 1
            if p2_last_pkmn and p2_pkmn_name != p2_last_pkmn:
                p2_feats['switches'] += 1
                p2_feats[f'switches_{w}'] += 1
                if prev_turn_damage_taken_p2 >= 15:
                    p2_feats['forced_switches'] += 1
            p1_last_pkmn = p1_pkmn_name
            p2_last_pkmn = p2_pkmn_name

            # --- STATUS INFLITTI (onset) ---
            p2_current_status = p2_state.get('status', 'nostatus') or 'nostatus'
            if p2_pkmn_name and p2_current_status != 'nostatus' and p1_opponent_last_status.get(p2_pkmn_name, 'nostatus') == 'nostatus':
                p1_feats['status_inflicted'] += 1
                p1_feats[f'status_inflicted_{w}'] += 1
            if p2_pkmn_name:
                p1_opponent_last_status[p2_pkmn_name] = p2_current_status

            p1_current_status = p1_state.get('status', 'nostatus') or 'nostatus'
            if p1_pkmn_name and p1_current_status != 'nostatus' and p2_opponent_last_status.get(p1_pkmn_name, 'nostatus') == 'nostatus':
                p2_feats['status_inflicted'] += 1
                p2_feats[f'status_inflicted_{w}'] += 1
            if p1_pkmn_name:
                p2_opponent_last_status[p1_pkmn_name] = p1_current_status

            # --- STATUS PERSISTENCE (turni sotto status) ---
            if p2_current_status != 'nostatus':
                p1_feats['status_turns_inflicted'] += 1
            if p1_current_status != 'nostatus':
                p2_feats['status_turns_inflicted'] += 1

            # --- MOSSE PRIORITARIE ---
            if p1_move and p1_move.get('priority', 0) > 0:
                p1_feats['priority_moves'] += 1
                p1_feats[f'priority_moves_{w}'] += 1
            if p2_move and p2_move.get('priority', 0) > 0:
                p2_feats['priority_moves'] += 1
                p2_feats[f'priority_moves_{w}'] += 1

            # --- Danno inflitto e faint ---
            # Danno P1->P2
            # hp_pct normalizzato a [0,100]
            def _hp(x):
                try:
                    v = float(x)
                except (ValueError, TypeError):
                    v = 0.0
                return max(0.0, min(100.0, v))

            p2_hp = _hp(p2_state.get('hp_pct', 0))
            damage_p1 = 0
            if p2_pkmn_name and p2_pkmn_name in p1_last_hp:
                damage_p1 = p1_last_hp[p2_pkmn_name] - p2_hp
                if damage_p1 > 0:
                    p1_feats['damage_dealt'] += damage_p1
                    p1_feats[f'damage_dealt_{w}'] += damage_p1
            if p2_pkmn_name:
                p1_last_hp[p2_pkmn_name] = p2_hp
            if p2_hp == 0:
                p1_feats['fainted_pokemon'] += 1
                p1_feats[f'fainted_pokemon_{w}'] += 1
                if p1_feats['time_to_first_ko_inflicted'] == -1:
                    p1_feats['time_to_first_ko_inflicted'] = t_idx + 1

            # EMA danno P1
            p1_feats['ema_damage_dealt'] = alpha * max(damage_p1, 0) + (1 - alpha) * p1_feats['ema_damage_dealt']

            # Danno P2->P1
            p1_hp = _hp(p1_state.get('hp_pct', 0))
            damage_p2 = 0
            if p1_pkmn_name and p1_pkmn_name in p2_last_hp:
                damage_p2 = p2_last_hp[p1_pkmn_name] - p1_hp
                if damage_p2 > 0:
                    p2_feats['damage_dealt'] += damage_p2
                    p2_feats[f'damage_dealt_{w}'] += damage_p2
            if p1_pkmn_name:
                p2_last_hp[p1_pkmn_name] = p1_hp
            if p1_hp == 0:
                p2_feats['fainted_pokemon'] += 1
                p2_feats[f'fainted_pokemon_{w}'] += 1
                if p2_feats['time_to_first_ko_inflicted'] == -1:
                    p2_feats['time_to_first_ko_inflicted'] = t_idx + 1

            # EMA danno P2
            p2_feats['ema_damage_dealt'] = alpha * max(damage_p2, 0) + (1 - alpha) * p2_feats['ema_damage_dealt']

            # --- BOOST (somma totale + per-stat positiva) ---
            p1_boosts = p1_state.get('boosts', {}) or {}
            p2_boosts = p2_state.get('boosts', {}) or {}
            p1_boosts_sum = sum(v for v in p1_boosts.values() if isinstance(v, (int, float)) and v > 0)
            p2_boosts_sum = sum(v for v in p2_boosts.values() if isinstance(v, (int, float)) and v > 0)
            p1_feats['boosts'] += p1_boosts_sum
            p1_feats[f'boosts_{w}'] += p1_boosts_sum
            p2_feats['boosts'] += p2_boosts_sum
            p2_feats[f'boosts_{w}'] += p2_boosts_sum
            for stat in ['atk', 'def', 'spa', 'spd', 'spe']:
                val1 = p1_boosts.get(stat, 0)
                if isinstance(val1, (int, float)) and val1 > 0:
                    p1_feats[f'boosts_{stat}_sum'] += val1
                val2 = p2_boosts.get(stat, 0)
                if isinstance(val2, (int, float)) and val2 > 0:
                    p2_feats[f'boosts_{stat}_sum'] += val2

            # Lead changes basati su danno cumulativo (semplice proxy di controllo del match)
            p1_cum_damage += max(damage_p1, 0)
            p2_cum_damage += max(damage_p2, 0)
            leader = 0
            if p1_cum_damage > p2_cum_damage:
                leader = 1
            elif p2_cum_damage > p1_cum_damage:
                leader = -1
            if last_leader != 0 and leader != 0 and leader != last_leader:
                p1_feats['lead_changes'] += 1
                p2_feats['lead_changes'] += 1
            if leader != 0:
                last_leader = leader

            # Memorizza danni subiti nel turno per rilevare switch forzati al turno successivo
            prev_turn_damage_taken_p2 = max(damage_p1, 0)
            prev_turn_damage_taken_p1 = max(damage_p2, 0)

    # Rate per turno (normalizziamo i conteggi per numero di turni considerati)
    if n_turns > 0:
        for m in base_metrics:
            p1_feats[f'{m}_per_turn'] = p1_feats[m] / n_turns
            p2_feats[f'{m}_per_turn'] = p2_feats[m] / n_turns
    else:
        for m in base_metrics:
            p1_feats[f'{m}_per_turn'] = 0
            p2_feats[f'{m}_per_turn'] = 0

    return p1_feats, p2_feats

def process_battle(battle_json, max_turns=30):
    """Processa un singolo JSON di battaglia in un dizionario di feature pronte per il modello.

    Cosa fa in pratica:
    - costruisce feature statiche di p1 e p2 dai team (in test p2 può essere solo il lead);
    - costruisce feature dinamiche usando i primi `max_turns` turni;
    - normalizza la prospettiva: p1_* deve rappresentare sempre il "player" (player_won si riferisce a p1_*);
    - crea feature differenza diff_* = p1_* − p2_*;
    - aggiunge un vantaggio type-chart per i lead e una proxy di outspeed con base_spe.

    Ritorna: dict con tutte le feature della battaglia (più battle_id e player_won se presente).
    """
    record = {'battle_id': battle_json['battle_id']}
    
    if 'player_won' in battle_json:
        record['player_won'] = battle_json['player_won']
    
    # Estrai feature statiche (in test p2 potrebbe avere solo il lead)
    p1_static = get_static_features(battle_json.get('p1_team_details', []), 'p1')
    p2_team = battle_json.get('p2_team_details')
    if not p2_team:
        lead = battle_json.get('p2_lead_details') or {}
        p2_team = [lead] if lead else []
    p2_static = get_static_features(p2_team, 'p2')
    # Estrai feature dinamiche (in test il log può chiamarsi 'battle_timeline')
    timeline = battle_json.get('battle_log')
    if timeline is None:
        timeline = battle_json.get('battle_timeline', [])
    p1_dynamic, p2_dynamic = get_dynamic_features(timeline, max_turns=max_turns)

    # Normalizza la prospettiva: p1_* deve sempre rappresentare il "player" (colui a cui si riferisce player_won)
    player_is_p1 = True
    if 'player_is_p1' in battle_json:
        try:
            player_is_p1 = bool(battle_json.get('player_is_p1'))
        except (ValueError, TypeError):
            player_is_p1 = True
    elif 'player_side' in battle_json:
        player_is_p1 = (str(battle_json.get('player_side')).lower() == 'p1')

    if not player_is_p1:
        # Scambia p1/p2 statiche e dinamiche per allineare p1_ al player
        p1_static, p2_static = p2_static, p1_static
        p1_dynamic, p2_dynamic = p2_dynamic, p1_dynamic
    p1_dynamic_renamed = {f'p1_{k}': v for k, v in p1_dynamic.items()}
    p2_dynamic_renamed = {f'p2_{k}': v for k, v in p2_dynamic.items()}
    
    record.update(p1_static)
    record.update(p2_static)
    record.update(p1_dynamic_renamed)
    record.update(p2_dynamic_renamed)
    
    all_p1_keys = list(p1_static.keys()) + list(p1_dynamic_renamed.keys())
    for p1_key in all_p1_keys:
        p2_key = p1_key.replace('p1_', 'p2_')
        diff_key = p1_key.replace('p1_', 'diff_')
        record[diff_key] = record.get(p1_key, 0) - record.get(p2_key, 0)

    # --- Type-chart advantage esplicito ---
    # Stimiamo per il lead: p1 lead vs p2 lead, se disponibili; fallback su media di team types.
    def _team_types(team):
        t = []
        for p in team or []:
            t.extend(p.get('types', []) or [])
        return list(set([tt for tt in t if tt and tt != 'notype']))

    p1_lead = battle_json.get('p1_lead_details') or (battle_json.get('battle_timeline', [{}])[0].get('p1_pokemon_state') if battle_json.get('battle_timeline') else None)
    p2_lead = battle_json.get('p2_lead_details') or (battle_json.get('battle_timeline', [{}])[0].get('p2_pokemon_state') if battle_json.get('battle_timeline') else None)
    p1_types_lead = (p1_lead.get('types') if isinstance(p1_lead, dict) else []) or []
    p2_types_lead = (p2_lead.get('types') if isinstance(p2_lead, dict) else []) or []

    if not p1_types_lead:
        p1_types_lead = _team_types(battle_json.get('p1_team_details', []))
    if not p2_types_lead:
        p2_types_lead = _team_types(p2_team)

    p1_off_mult = type_multiplier(p1_types_lead, p2_types_lead)
    p2_off_mult = type_multiplier(p2_types_lead, p1_types_lead)
    record['p1_type_off_mult'] = p1_off_mult
    record['p2_type_off_mult'] = p2_off_mult
    record['diff_type_off_mult'] = p1_off_mult - p2_off_mult

    # --- Lead matchup & speed control ---
    p1_lead_spe = 0
    p2_lead_spe = 0
    if isinstance(p1_lead, dict):
        p1_lead_spe = float(p1_lead.get('base_spe', 0))
    if isinstance(p2_lead, dict):
        p2_lead_spe = float(p2_lead.get('base_spe', 0))
    record['p1_lead_base_spe'] = p1_lead_spe
    record['p2_lead_base_spe'] = p2_lead_spe
    record['diff_lead_base_spe'] = p1_lead_spe - p2_lead_spe
    # Probabilità grezza di outspeed: sigmoide della differenza
    record['p1_outspeed_prob'] = 1.0 / (1.0 + np.exp(-0.05 * (p1_lead_spe - p2_lead_spe)))
    record['diff_outspeed_prob'] = record['p1_outspeed_prob'] - (1.0 - record['p1_outspeed_prob'])
        
    return record

def create_feature_df(file_path, max_turns=30):
    """Legge un file .jsonl (una battaglia per riga) e crea un DataFrame con le feature.

    Parametri:
    - file_path: percorso al .jsonl
    - max_turns: limite turni per le feature dinamiche

    Ritorna: pandas.DataFrame con una riga per battle_id e tutte le colonne di feature.
    """
    battles_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Processing {file_path.split('/')[-1]}"):
            battle = json.loads(line)
            battles_data.append(process_battle(battle, max_turns=max_turns))
            
    return pd.DataFrame(battles_data)