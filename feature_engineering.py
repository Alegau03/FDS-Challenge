"""
Pokemon Battle Predictor - Feature Engineering Module

Questo modulo genera feature per predire il vincitore di battaglie Pokemon competitive.
Le feature catturano composizione team, dinamiche di battaglia e interazioni strategiche.

Architettura Feature:
    1. STATIC FEATURES (Team Composition):
       - Stats base Pokemon (HP, Atk, Def, SpA, SpD, Spe)
       - Copertura offensive/defensive per tipo
       - Bilanciamento team offense/defense
       - Predizioni tipo Pokemon (usando Type Distribution da predict.csv)
    
    2. DYNAMIC FEATURES (Battle Log):
       - Damage dealt/taken per finestra temporale (w1, w2, w3)
       - KO, status, switch tracking
       - Super effective hits e type advantages
       - HP tracking (start, end, delta)
       - Momentum e pressure offensive
    
    3. INTERACTION FEATURES:
       - Prodotti tra feature correlate (es. damage × speed)
       - Combinazioni non-lineari (es. switch × HP advantage)

Input:
    - File JSONL con battaglie formato Showdown (train.jsonl / test.jsonl)
    - predict.csv per Type Distribution (probabilità tipo per Pokemon)

Output:
    - DataFrame con battle_id, player_won (train) e 339 feature
"""

import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from predictor import load_type_distribution, predict_unseen_types

# Carica distribuzione tipi (globale, caricato una volta)
TYPE_DISTRIBUTION_DF = None

# --- Type chart (semplificato): efficacia attacco -> moltiplicatore ---
# Nota: per brevità includiamo subset dei tipi più comuni; i tipi mancanti saranno trattati come 1.0 (neutro).
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

# helper per lower safe
_def_not = 'notype'

def _safe_lower(x):
    return (x or '').lower()

def _get_num(x, default=0.0):
    """Safe float conversion."""
    try:
        if x is None:
            return float(default)
        return float(x)
    except (TypeError, ValueError):
        return float(default)

def type_multiplier(attacking_types, defending_types):
    mult = 1.0
    for atk in attacking_types or []:
        atk = _safe_lower(atk)
        row = TYPE_CHART.get(atk, {})
        for df in defending_types or []:
            df = _safe_lower(df)
            mult *= float(row.get(df, 1.0))
    return mult

def best_stab_multiplier(attacking_types, defending_types):
    """Massimo moltiplicatore STAB possibile dato i tipi attaccanti vs difensori."""
    best = 1.0
    atks = [t for t in (attacking_types or []) if t and _safe_lower(t) != _def_not]
    defs = [t for t in (defending_types or []) if t and _safe_lower(t) != _def_not]
    if not atks or not defs:
        return 1.0
    for atk in atks:
        row = TYPE_CHART.get(_safe_lower(atk), {})
        m = 1.0
        for df in defs:
            m *= float(row.get(_safe_lower(df), 1.0))
        if m > best:
            best = m
    return float(best)

def effectiveness_multiplier(move_type, defending_types):
    """Moltiplicatore di efficacia per una singola mossa (move_type) contro i tipi difensivi.
    Se il tipo non è nel TYPE_CHART o non disponibile, ritorna 1.0.
    """
    if not move_type:
        return 1.0
    mt = _safe_lower(move_type)
    row = TYPE_CHART.get(mt, {})
    mult = 1.0
    for df in defending_types or []:
        mult *= float(row.get(_safe_lower(df), 1.0))
    return float(mult)

def compute_expected_type_advantage(p1_types: list, p2_seen_types: list, type_distribution_df: pd.DataFrame) -> float:
    """
    Calcola il vantaggio di tipo atteso di P1 contro i tipi NON visti di P2.
    
    Logica 
    1. Usa predict_unseen_types() per ottenere probabilità di ogni tipo non visto
    2. Per ogni tipo non visto, calcola il moltiplicatore di tipo di P1 contro quel tipo
    3. Calcola media pesata per probabilità: sum(prob_tipo * moltiplicatore_tipo)
    
    Args:
        p1_types: Lista dei tipi del team P1 (possono essere ripetuti se più pokemon dello stesso tipo)
        p2_seen_types: Lista dei tipi visti di P2 durante la battaglia
        type_distribution_df: DataFrame con distribuzione dei tipi (da predict.csv)
    
    Returns:
        Float: vantaggio di tipo atteso (media pesata dei moltiplicatori)
    """
    if not p1_types or not type_distribution_df is not None:
        return 1.0  # Neutro se non abbiamo info
    
    # Predici tipi non visti con le loro probabilità
    unseen_type_probs = predict_unseen_types(
        seen_types=p2_seen_types,
        type_distribution=type_distribution_df,
        team_size=6
    )
    
    if not unseen_type_probs:
        return 1.0  # Nessun tipo da predire
    
    # Calcola vantaggio atteso
    expected_advantage = 0.0
    total_prob = 0.0
    
    for unseen_type, prob in unseen_type_probs.items():
        # Per ogni tipo non visto, calcola il moltiplicatore medio di P1 contro quel tipo
        # Usiamo il miglior tipo di P1 contro questo tipo
        best_mult = 0.0
        for p1_type in set(p1_types):  # usa set per evitare ripetizioni
            mult = type_multiplier([p1_type], [unseen_type])
            if mult > best_mult:
                best_mult = mult
        
        # Pesa per la probabilità
        expected_advantage += prob * best_mult
        total_prob += prob
    
    # Normalizza
    if total_prob > 0:
        expected_advantage /= total_prob
    
    return float(expected_advantage)

def get_static_features(team_details, prefix):
    """Estrae feature statiche basate sulla composizione del team."""
    features = {}
    stats = ['base_hp', 'base_atk', 'base_def', 'base_spa', 'base_spd', 'base_spe']
    
    all_stats_values = {stat: [] for stat in stats}
    all_types = []
    offense_vals = []
    defense_vals = []
    speed_vals = []
    bst_vals = []

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
            bst_vals.append(hp + atk + de + spa + sd + spe)

    for stat, values in all_stats_values.items():
        if values:
            features[f'{prefix}_team_mean_{stat}'] = np.mean(values)
            features[f'{prefix}_team_std_{stat}'] = np.std(values)
        else:
            features[f'{prefix}_team_mean_{stat}'] = 0
            features[f'{prefix}_team_std_{stat}'] = 0

    # HP totale del team (somma base HP)
    features[f'{prefix}_team_total_base_hp'] = sum(all_stats_values['base_hp']) if all_stats_values['base_hp'] else 0

    features[f'{prefix}_team_unique_types'] = len(set(t for t in all_types if t != 'notype'))

    # Indici compositi aggregati
    if offense_vals:
        features[f'{prefix}_team_mean_offense'] = float(np.mean(offense_vals))
        features[f'{prefix}_team_std_offense'] = float(np.std(offense_vals))
    else:
        features[f'{prefix}_team_mean_offense'] = 0.0
        features[f'{prefix}_team_std_offense'] = 0.0
    if defense_vals:
        features[f'{prefix}_team_mean_defense'] = float(np.mean(defense_vals))
        features[f'{prefix}_team_std_defense'] = float(np.std(defense_vals))
    else:
        features[f'{prefix}_team_mean_defense'] = 0.0
        features[f'{prefix}_team_std_defense'] = 0.0
    if speed_vals:
        features[f'{prefix}_team_mean_speed'] = float(np.mean(speed_vals))
        features[f'{prefix}_team_std_speed'] = float(np.std(speed_vals))
    else:
        features[f'{prefix}_team_mean_speed'] = 0.0
        features[f'{prefix}_team_std_speed'] = 0.0

    # BST aggregati
    if bst_vals:
        features[f'{prefix}_team_mean_bst'] = float(np.mean(bst_vals))
        features[f'{prefix}_team_std_bst'] = float(np.std(bst_vals))
    else:
        features[f'{prefix}_team_mean_bst'] = 0.0
        features[f'{prefix}_team_std_bst'] = 0.0

    # Rapporto offense/defense
    denom = features[f'{prefix}_team_mean_defense'] if features[f'{prefix}_team_mean_defense'] != 0 else 1.0
    features[f'{prefix}_team_offense_defense_ratio'] = features[f'{prefix}_team_mean_offense'] / denom

    # Entropia dei tipi
    types = [t for t in all_types if t != 'notype']
    if types:
        values, counts = np.unique(types, return_counts=True)
        probs = counts / counts.sum()
        type_entropy = float(-(probs * np.log(probs + 1e-12)).sum())
        max_entropy = float(np.log(len(values))) if len(values) > 0 else 1.0
        norm_entropy = float(type_entropy / (max_entropy + 1e-12)) if max_entropy > 0 else 0.0
    else:
        type_entropy = 0.0
        norm_entropy = 0.0
    features[f'{prefix}_team_type_entropy'] = type_entropy
    features[f'{prefix}_team_type_entropy_norm'] = norm_entropy
    
    # coverage_gaps RIMOSSO - feature statica, poco predittiva (team può vincere anche con gap)
    
    return features

def get_dynamic_features(battle_log, max_turns=30, p1_team_total_hp=0, p2_team_total_hp=0, 
                        p1_team_details=None, p2_team_details=None):
    #Estrae feature dinamiche per entrambi i giocatori basandosi sul battle log.

    # Inizializza le feature base
    base_metrics = ['damage_dealt', 'fainted_pokemon', 'switches', 'status_inflicted']  # V6: rimossi boosts, priority_moves

    p1_pokemon_types = {}
    p2_pokemon_types = {}
    if p1_team_details:
        for pkmn in p1_team_details:
            name = _safe_lower(pkmn.get('name', ''))
            types = [_safe_lower(t) for t in (pkmn.get('types', []) or []) if t]
            if name:
                p1_pokemon_types[name] = types
    if p2_team_details:
        for pkmn in p2_team_details:
            name = _safe_lower(pkmn.get('name', ''))
            types = [_safe_lower(t) for t in (pkmn.get('types', []) or []) if t]
            if name:
                p2_pokemon_types[name] = types


    def _init_move_rich(d):
        # Aggiunge al dict d i campi per feature mosse avanzate.
        # conteggi generali mosse
        d['moves_used'] = 0
        d['damaging_moves_used'] = 0
        d['status_moves_used'] = 0
        d['phys_moves_count'] = 0
        d['spec_moves_count'] = 0
        # danni per categoria
        d['phys_damage'] = 0.0
        d['spec_damage'] = 0.0
        # STAB
        d['stab_moves_count'] = 0
        d['stab_damage'] = 0.0
        d['nonstab_moves_count'] = 0
        d['nonstab_damage'] = 0.0
        # efficacia
        d['super_effective_hits'] = 0
        d['notvery_effective_hits'] = 0
        d['avg_effectiveness_used'] = 1.0
        d['max_effectiveness_used'] = 1.0
        d['min_effectiveness_used'] = 1.0
        d['sum_effectiveness_used'] = 0.0
        # accuratezza/crit/miss
        d['total_hits'] = 0  # somma dei colpi (multi-hit inclusi)
        d['sum_base_power'] = 0.0
        d['sum_accuracy'] = 0.0
        # pressione fisica/speciale (offense vs defense)
        return d
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
        # dinamiche aggiuntive
        d['net_damage_diff'] = 0.0
        d['net_damage_diff_w1'] = 0.0
        d['net_damage_diff_w2'] = 0.0
        d['net_damage_diff_w3'] = 0.0
        d['momentum_positive_turns'] = 0
        # HP trajectory (media HP attivo per turno)
        d['avg_hp_pct_start'] = 0.0
        d['avg_hp_pct_end'] = 0.0
        d['avg_hp_pct_delta'] = 0.0
        # status per tipo (onset)
        for st in ['brn', 'psn', 'tox', 'par', 'slp', 'frz', 'conf']:
            d[f'status_inflicted_{st}'] = 0
        
        d['team_hp_remaining_pct'] = 0.0
        # Serve per calcolare la percentuale di HP rimanente del team
        d['avg_hp_per_alive'] = 0.0
        # condizioni di campo basilari, se disponibili
        # arricchimento mosse
        _init_move_rich(d)
        return d

    p1_feats = init_feat_dict()
    p2_feats = init_feat_dict()

    p1_last_hp = {}
    p2_last_hp = {}

    # Tracking HP per pokemon (per calcolare metriche team HP)
    p1_pokemon_hp = {}  # {pokemon_name: hp_pct}
    p2_pokemon_hp = {}  # {pokemon_name: hp_pct}


    # Variabili per tracciare lo stato precedente
    p1_last_pkmn = None
    p2_last_pkmn = None
    p1_opponent_last_status = {}
    p2_opponent_last_status = {}

    if not battle_log:
        return p1_feats, p2_feats

    # Parametri per EMA
    alpha = 0.3  # recency weighting

    # Limita ai primi `max_turns` turni
    n_turns = min(max_turns, len(battle_log))
    # Variabili per lead e forced switches
    p1_cum_damage = 0.0
    p2_cum_damage = 0.0
    last_leader = 0  # 1 se p1 in vantaggio, -1 se p2, 0 pari
    prev_turn_damage_taken_p1 = 0.0
    prev_turn_damage_taken_p2 = 0.0

    # serie HP per trajectory
    p1_hp_series = []
    p2_hp_series = []

    def _calc_team_hp_metrics(pokemon_hp_dict, team_total_base_hp):
        if not pokemon_hp_dict or team_total_base_hp == 0:
            return {
                'team_hp_remaining_pct': 0.0,
                'avg_hp_per_alive': 0.0
            }
        
        hp_values = list(pokemon_hp_dict.values())
        alive_hp = [h for h in hp_values if h > 0]
        
        # % HP rimanente del team (media HP% dei pokemon)
        avg_hp_pct = np.mean(hp_values) if hp_values else 0.0
        
        # Media HP per pokemon vivo
        avg_per_alive = np.mean(alive_hp) if alive_hp else 0.0
        
        return {
            'team_hp_remaining_pct': float(avg_hp_pct),
            'avg_hp_per_alive': float(avg_per_alive)
        }

    for t_idx, turn in enumerate(battle_log[:max_turns]):
        # Identifica finestra
        if t_idx < 10:
            w = 'w1'
        elif t_idx < 20:
            w = 'w2'
        else:
            w = 'w3'

        p1_state = turn.get('p1_pokemon_state', {}) or {}
        p2_state = turn.get('p2_pokemon_state', {}) or {}
        p1_move = turn.get('p1_move_details', {}) or {}
        p2_move = turn.get('p2_move_details', {}) or {}
            
        if p1_state and p2_state:
            p1_pkmn_name = p1_state.get('name')
            p2_pkmn_name = p2_state.get('name')

            # --- SWITCH ---
            if p1_last_pkmn and p1_pkmn_name != p1_last_pkmn:
                p1_feats['switches'] += 1
                p1_feats[f'switches_{w}'] += 1
            if p2_last_pkmn and p2_pkmn_name != p2_last_pkmn:
                p2_feats['switches'] += 1
                p2_feats[f'switches_{w}'] += 1
            p1_last_pkmn = p1_pkmn_name
            p2_last_pkmn = p2_pkmn_name

            # --- STATUS INFLITTI (onset) ---
            p2_current_status = p2_state.get('status', 'nostatus') or 'nostatus'
            if p2_pkmn_name and p2_current_status != 'nostatus' and p1_opponent_last_status.get(p2_pkmn_name, 'nostatus') == 'nostatus':
                p1_feats['status_inflicted'] += 1
                p1_feats[f'status_inflicted_{w}'] += 1
                st = _safe_lower(p2_current_status)
                if st in ['brn', 'psn', 'tox', 'par', 'slp', 'frz', 'conf']:
                    p1_feats[f'status_inflicted_{st}'] += 1
            if p2_pkmn_name:
                p1_opponent_last_status[p2_pkmn_name] = p2_current_status

            p1_current_status = p1_state.get('status', 'nostatus') or 'nostatus'
            if p1_pkmn_name and p1_current_status != 'nostatus' and p2_opponent_last_status.get(p1_pkmn_name, 'nostatus') == 'nostatus':
                p2_feats['status_inflicted'] += 1
                p2_feats[f'status_inflicted_{w}'] += 1
                st = _safe_lower(p1_current_status)
                if st in ['brn', 'psn', 'tox', 'par', 'slp', 'frz', 'conf']:
                    p2_feats[f'status_inflicted_{st}'] += 1
            if p1_pkmn_name:
                p2_opponent_last_status[p1_pkmn_name] = p1_current_status

            # --- STATUS PERSISTENCE (turni sotto status) ---
            if p2_current_status != 'nostatus':
                p1_feats['status_turns_inflicted'] += 1
            if p1_current_status != 'nostatus':
                p2_feats['status_turns_inflicted'] += 1
            # --- Danno inflitto e faint ---
            # hp_pct normalizzato a [0,100]
            def _hp(x):
                try:
                    v = float(x)
                except (TypeError, ValueError):
                    v = 0.0
                return max(0.0, min(100.0, v))

            # Danno P1->P2
            p2_hp = _hp(p2_state.get('hp_pct', 0))
            damage_p1 = 0
            if p2_pkmn_name and p2_pkmn_name in p1_last_hp:
                damage_p1 = p1_last_hp[p2_pkmn_name] - p2_hp
                if damage_p1 > 0:
                    p1_feats['damage_dealt'] += damage_p1
                    p1_feats[f'damage_dealt_{w}'] += damage_p1
            if p2_pkmn_name:
                p1_last_hp[p2_pkmn_name] = p2_hp
                p2_pokemon_hp[p2_pkmn_name] = p2_hp  # Track per metriche team HP
            
            if p2_hp == 0:
                p1_feats['fainted_pokemon'] += 1
                p1_feats[f'fainted_pokemon_{w}'] += 1

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
                p1_pokemon_hp[p1_pkmn_name] = p1_hp  # Track per metriche team HP
            
            if p1_hp == 0:
                p2_feats['fainted_pokemon'] += 1
                p2_feats[f'fainted_pokemon_{w}'] += 1

            # Aggiungi a serie HP per trajectory
            p1_hp_series.append(p1_hp)
            p2_hp_series.append(p2_hp)

            # EMA danno P2
            p2_feats['ema_damage_dealt'] = alpha * max(damage_p2, 0) + (1 - alpha) * p2_feats['ema_damage_dealt']

            # --- BOOST (somma totale + per-stat positiva) ---
            p1_boosts = p1_state.get('boosts', {}) or {}
            p2_boosts = p2_state.get('boosts', {}) or {}
            for stat in ['atk', 'def', 'spa', 'spd', 'spe']:
                val1 = p1_boosts.get(stat, 0)
            # --- Feature ricche sulle mosse, per P1 e P2 ---
            def update_move_features(feat_dict, move_dict, atk_state, def_state, observed_damage, window_tag, 
                                    atk_pokemon_types_lookup, def_pokemon_types_lookup):
                if not isinstance(move_dict, dict) or not move_dict:
                    return
                feat_dict['moves_used'] += 1
                move_type = _safe_lower(move_dict.get('type'))
                category = _safe_lower(move_dict.get('category'))  # 'physical' / 'special' / 'status'
                base_power = _get_num(move_dict.get('base_power') or move_dict.get('basePower') or move_dict.get('power') or 0.0)
                accuracy = _get_num(move_dict.get('accuracy') or move_dict.get('acc') or 0.0)
                hits = int(_get_num(move_dict.get('hits') or 1, 1))
                is_crit = bool(move_dict.get('is_crit') or move_dict.get('crit') or False)
                did_hit = move_dict.get('hit')
                feat_dict['total_hits'] += max(1, hits)
                feat_dict['sum_base_power'] += base_power
                feat_dict['sum_accuracy'] += accuracy if accuracy > 0 else 0.0

                atk_name = _safe_lower(atk_state.get('name', ''))
                def_name = _safe_lower(def_state.get('name', ''))
                atk_types = atk_pokemon_types_lookup.get(atk_name, [])
                def_types = def_pokemon_types_lookup.get(def_name, [])
                is_stab = move_type in atk_types if move_type else False

                # Efficacia
                eff = effectiveness_multiplier(move_type, def_types) if move_type else 1.0
                feat_dict['sum_effectiveness_used'] += eff
                feat_dict['max_effectiveness_used'] = max(feat_dict['max_effectiveness_used'], eff)
                feat_dict['min_effectiveness_used'] = min(feat_dict['min_effectiveness_used'], eff) if feat_dict['moves_used'] > 1 else eff
                if eff > 1.0:
                    feat_dict['super_effective_hits'] += 1
                elif 0.0 < eff < 1.0:
                    feat_dict['notvery_effective_hits'] += 1

                # Danni osservati (da HP delta)
                od = max(0.0, _get_num(observed_damage))
                if category in ['physical','special']:
                    feat_dict['damaging_moves_used'] += 1
                else:
                    feat_dict['status_moves_used'] += 1

                # split per categoria
                if category == 'physical':
                    feat_dict['phys_moves_count'] += 1
                    feat_dict['phys_damage'] += od
                    # pressione: att/def * (base_power o od)
                    att = _get_num(atk_state.get('base_atk'), 0.0)
                    de = max(1.0, _get_num(def_state.get('base_def'), 1.0))
                elif category == 'special':
                    feat_dict['spec_moves_count'] += 1
                    feat_dict['spec_damage'] += od
                    spa = _get_num(atk_state.get('base_spa'), 0.0)
                    spd = max(1.0, _get_num(def_state.get('base_spd'), 1.0))
                # STAB aggregati
                if is_stab:
                    feat_dict['stab_moves_count'] += 1
                    feat_dict['stab_damage'] += od
                else:
                    feat_dict['nonstab_moves_count'] += 1
                    feat_dict['nonstab_damage'] += od
            # aggiorna p1 con la sua mossa contro p2
            update_move_features(p1_feats, p1_move, p1_state, p2_state, damage_p1, w, 
                                p1_pokemon_types, p2_pokemon_types)
            # aggiorna p2 con la sua mossa contro p1
            update_move_features(p2_feats, p2_move, p2_state, p1_state, damage_p2, w, 
                                p2_pokemon_types, p1_pokemon_types)

            # Lead changes basati su danno cumulativo
            p1_cum_damage += max(damage_p1, 0)
            p2_cum_damage += max(damage_p2, 0)
            # net damage diff e momentum (per prospettiva di ciascun lato)
            p1_feats['net_damage_diff'] += max(damage_p1, 0) - max(damage_p2, 0)
            p2_feats['net_damage_diff'] += max(damage_p2, 0) - max(damage_p1, 0)
            if (max(damage_p1, 0) - max(damage_p2, 0)) > 0:
                p1_feats['momentum_positive_turns'] += 1
            if (max(damage_p2, 0) - max(damage_p1, 0)) > 0:
                p2_feats['momentum_positive_turns'] += 1
            p1_feats[f'net_damage_diff_{w}'] += max(damage_p1, 0) - max(damage_p2, 0)
            p2_feats[f'net_damage_diff_{w}'] += max(damage_p2, 0) - max(damage_p1, 0)

            leader = 0
            if p1_cum_damage > p2_cum_damage:
                leader = 1
            elif p2_cum_damage > p1_cum_damage:
                leader = -1
            if leader != 0:
                last_leader = leader

            # Memorizza danni subiti nel turno per rilevare switch forzati al turno successivo
            prev_turn_damage_taken_p2 = max(damage_p1, 0)
            prev_turn_damage_taken_p1 = max(damage_p2, 0)

    # HP trajectory sintetica
    def _traj(series):
        if not series:
            return (0.0, 0.0, 0.0)
        k = min(3, len(series))
        start = float(np.mean(series[:k]))
        end = float(np.mean(series[-k:]))
        return (start, end, end - start)
    
    s, e, d = _traj(p1_hp_series)
    p1_feats['avg_hp_pct_start'] = s
    p1_feats['avg_hp_pct_end'] = e
    p1_feats['avg_hp_pct_delta'] = d
    s, e, d = _traj(p2_hp_series)
    p2_feats['avg_hp_pct_start'] = s
    p2_feats['avg_hp_pct_end'] = e
    p2_feats['avg_hp_pct_delta'] = d

    p1_hp_metrics = _calc_team_hp_metrics(p1_pokemon_hp, p1_team_total_hp)
    p2_hp_metrics = _calc_team_hp_metrics(p2_pokemon_hp, p2_team_total_hp)
    p1_feats.update(p1_hp_metrics)
    p2_feats.update(p2_hp_metrics)


    p1_ko_eff = p2_feats.get('fainted_pokemon', 0) / max(1, p1_feats.get('fainted_pokemon', 1))
    p2_ko_eff = p1_feats.get('fainted_pokemon', 0) / max(1, p2_feats.get('fainted_pokemon', 1))
    p1_feats['ko_efficiency'] = float(p1_ko_eff)
    p2_feats['ko_efficiency'] = float(p2_ko_eff)

    # Rate per turno
    if n_turns > 0:
        for m in base_metrics:
            p1_feats[f'{m}_per_turn'] = p1_feats[m] / n_turns
            p2_feats[f'{m}_per_turn'] = p2_feats[m] / n_turns
        # rate per-turn per alcune feature ricche
        for k in ['phys_damage','spec_damage','stab_damage','nonstab_damage']:
            p1_feats[f'{k}_per_turn'] = p1_feats.get(k, 0.0) / n_turns if n_turns else 0.0
            p2_feats[f'{k}_per_turn'] = p2_feats.get(k, 0.0) / n_turns if n_turns else 0.0
        # medie da somme e quote d'uso
        for side in (p1_feats, p2_feats):
            mv = max(1, int(side.get('moves_used', 0)))
            side['avg_base_power_used'] = side.get('sum_base_power', 0.0) / mv
            side['avg_accuracy_used'] = side.get('sum_accuracy', 0.0) / mv
            side['avg_effectiveness_used'] = side.get('sum_effectiveness_used', 0.0) / mv
            side['stab_rate'] = side.get('stab_moves_count', 0) / mv
            side['phys_move_rate'] = side.get('phys_moves_count', 0) / mv
            side['spec_move_rate'] = side.get('spec_moves_count', 0) / mv
            side['status_move_rate'] = side.get('status_moves_used', 0) / mv
            # share danni e tassi efficacia/crit/miss
            td = side.get('phys_damage', 0.0) + side.get('spec_damage', 0.0)
            side['phys_damage_share'] = (side.get('phys_damage', 0.0) / td) if td > 0 else 0.0
            side['spec_damage_share'] = (side.get('spec_damage', 0.0) / td) if td > 0 else 0.0
            dm = max(1, int(side.get('damaging_moves_used', 0)))
            side['super_effective_rate'] = side.get('super_effective_hits', 0) / dm
            side['notvery_effective_rate'] = side.get('notvery_effective_hits', 0) / dm
            th = max(1, int(side.get('total_hits', 0)))
            
    else:
        for m in base_metrics:
            p1_feats[f'{m}_per_turn'] = 0
            p2_feats[f'{m}_per_turn'] = 0
        for side in (p1_feats, p2_feats):
            side['avg_base_power_used'] = 0.0
            side['avg_accuracy_used'] = 0.0
            side['avg_effectiveness_used'] = 1.0
            side['stab_rate'] = 0.0
            side['phys_move_rate'] = 0.0
            side['spec_move_rate'] = 0.0
            side['status_move_rate'] = 0.0
            side['phys_damage_share'] = 0.0
            side['spec_damage_share'] = 0.0
            side['super_effective_rate'] = 0.0
            side['notvery_effective_rate'] = 0.0
            side['immune_rate'] = 0.0
    return p1_feats, p2_feats

def process_battle(battle_json, max_turns=30):
    """Processa un singolo JSON di una battaglia e restituisce un dizionario di feature.

    max_turns controlla la massima lunghezza del battle_log utilizzata per le feature dinamiche.
    """
    record = {'battle_id': battle_json['battle_id']}
    
    if 'player_won' in battle_json:
        record['player_won'] = battle_json['player_won']
    
    # Estrai feature statiche (in test p2 potrebbe avere solo il lead)
    p1_team = battle_json.get('p1_team_details', [])
    p1_static = get_static_features(p1_team, 'p1')
    p2_team = battle_json.get('p2_team_details')
    if not p2_team:
        lead = battle_json.get('p2_lead_details') or {}
        p2_team = [lead] if lead else []
    p2_static = get_static_features(p2_team, 'p2')
    # Estrai feature dinamiche (in test può essere 'battle_timeline')
    timeline = battle_json.get('battle_log')
    if timeline is None:
        timeline = battle_json.get('battle_timeline', [])
    p1_total_hp = p1_static.get('p1_team_total_base_hp', 0)
    p2_total_hp = p2_static.get('p2_team_total_base_hp', 0)
    p1_dynamic, p2_dynamic = get_dynamic_features(timeline, max_turns=max_turns, 
                                                   p1_team_total_hp=p1_total_hp, 
                                                   p2_team_total_hp=p2_total_hp,
                                                   p1_team_details=p1_team,
                                                   p2_team_details=p2_team)

    # Normalizza la prospettiva: p1_* deve sempre rappresentare il "player" (colui a cui si riferisce player_won)
    player_is_p1 = True
    if 'player_is_p1' in battle_json:
        val = battle_json.get('player_is_p1')
        if isinstance(val, bool):
            player_is_p1 = val
        elif isinstance(val, str):
            player_is_p1 = val.strip().lower() in ('true', '1', 'yes', 'y')
        else:
            player_is_p1 = bool(val)
    elif 'player_side' in battle_json:
        player_is_p1 = (str(battle_json.get('player_side')).lower() == 'p1')

    if not player_is_p1:
        # Scambia p1/p2 statiche e dinamiche per allineare p1_ al player
        p1_static, p2_static = p2_static, p1_static
        p1_dynamic, p2_dynamic = p2_dynamic, p1_dynamic

    team_a = p1_team if player_is_p1 else p2_team  # player
    team_b = p2_team if player_is_p1 else p1_team  # opponent
    if not team_a:
        la = battle_json.get('p1_lead_details') if player_is_p1 else battle_json.get('p2_lead_details')
        if la:
            team_a = [la]
    if not team_b:
        lb = battle_json.get('p2_lead_details') if player_is_p1 else battle_json.get('p1_lead_details')
        if lb:
            team_b = [lb]

    atk_pairs = []
    def_pairs = []
    for pa in team_a or []:
        for pb in team_b or []:
            atk_pairs.append(best_stab_multiplier(pa.get('types', []), pb.get('types', [])))
            def_pairs.append(best_stab_multiplier(pb.get('types', []), pa.get('types', [])))

    def _coverage_stats(arr):
        if not arr:
            return {
                'mean': 1.0, 'max': 1.0, 'min': 1.0,
                'super_count': 0, 'immune_count': 0, 'notvery_count': 0,
            }
        arr = np.array(arr, dtype=float)
        return {
            'mean': float(np.mean(arr)),
            'max': float(np.max(arr)),
            'min': float(np.min(arr)),
            'super_count': int(np.sum(arr > 1.0)),
            'immune_count': int(np.sum(arr == 0.0)),
            'notvery_count': int(np.sum((arr > 0.0) & (arr < 1.0))),
        }

    cov_a = _coverage_stats(atk_pairs)
    cov_b = _coverage_stats(def_pairs)
    p1_static.update({
        'p1_team_stab_cov_mean': cov_a['mean'],
        'p1_team_stab_cov_max': cov_a['max'],
        'p1_team_stab_cov_min': cov_a['min'],
        'p1_team_stab_super_count': cov_a['super_count'],
        'p1_team_stab_immune_count': cov_a['immune_count'],
        'p1_team_stab_notvery_count': cov_a['notvery_count'],
    })
    p2_static.update({
        'p2_team_stab_cov_mean': cov_b['mean'],
        'p2_team_stab_cov_max': cov_b['max'],
        'p2_team_stab_cov_min': cov_b['min'],
        'p2_team_stab_super_count': cov_b['super_count'],
        'p2_team_stab_immune_count': cov_b['immune_count'],
        'p2_team_stab_notvery_count': cov_b['notvery_count'],
    })

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

    # Estrai tipi visti di P2 dalla timeline
    p2_seen_types = set()
    if timeline:
        for turn in timeline[:max_turns]:
            p2_state = turn.get('p2_pokemon_state', {}) or {}
            if p2_state:
                p2_types = p2_state.get('types', []) or []
                for t in p2_types:
                    if t and t.lower() != 'notype':
                        p2_seen_types.add(t.lower())
    
    # Tipi di P1 (per calcolare vantaggio)
    p1_all_types = []
    for pkmn in (p1_team or []):
        p1_all_types.extend([t.lower() for t in pkmn.get('types', []) if t and t.lower() != 'notype'])
    
    # Calcola vantaggio atteso contro tipi non visti (usa distribuzione globale)
    global TYPE_DISTRIBUTION_DF
    expected_advantage = 1.0  # default neutro
    if TYPE_DISTRIBUTION_DF is not None and p1_all_types:
        expected_advantage = compute_expected_type_advantage(
            p1_types=p1_all_types,
            p2_seen_types=list(p2_seen_types),
            type_distribution_df=TYPE_DISTRIBUTION_DF
        )
    
    record['p1_expected_type_advantage_unseen_p2'] = expected_advantage
    return record

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge interaction features tra le feature più importanti.
    
    Le interaction features catturano relazioni non-lineari tra variabili
    che i GBDT potrebbero non scoprire facilmente da soli.
    
    Args:
        df: DataFrame con le feature base
        
    Returns:
        DataFrame con feature di interazione aggiunte
    """
    df = df.copy()
    
    # Danno × Velocità: chi fa più danno E è più veloce ha vantaggio
    if 'diff_damage_dealt' in df.columns and 'diff_lead_base_spe' in df.columns:
        df['interact_damage_x_speed'] = df['diff_damage_dealt'] * df['diff_lead_base_spe']
    
    # Bilancio team × Vantaggio tipi: team offensivo con vantaggio tipi è letale
    if 'p1_team_offense_defense_ratio' in df.columns and 'diff_type_off_mult' in df.columns:
        df['interact_p1_ratio_x_type'] = df['p1_team_offense_defense_ratio'] * df['diff_type_off_mult']
    if 'p2_team_offense_defense_ratio' in df.columns and 'diff_type_off_mult' in df.columns:
        df['interact_p2_ratio_x_type'] = df['p2_team_offense_defense_ratio'] * (-df['diff_type_off_mult'])

    # KO × Danno: pressure offensiva totale
    if 'diff_fainted_pokemon' in df.columns and 'diff_damage_dealt' in df.columns:
        df['interact_ko_x_damage'] = df['diff_fainted_pokemon'] * df['diff_damage_dealt']

    # Status × Danno: status debuffano quindi amplificano il danno
    if 'diff_status_inflicted' in df.columns and 'diff_damage_dealt' in df.columns:
        df['interact_status_x_damage'] = df['diff_status_inflicted'] * df['diff_damage_dealt']
    
    # Offense index × Type advantage
    if 'diff_team_mean_offense' in df.columns and 'diff_type_off_mult' in df.columns:
        df['interact_offense_x_type'] = df['diff_team_mean_offense'] * df['diff_type_off_mult']
    
    # Speed × Outspeed probability: velocità assoluta vs relativa
    if 'diff_lead_base_spe' in df.columns and 'diff_outspeed_prob' in df.columns:
        df['interact_speed_x_outspeed'] = df['diff_lead_base_spe'] * df['diff_outspeed_prob']
    
    # Rapporto offense/defense e danno per turno (w1)
    if 'p1_team_offense_defense_ratio' in df.columns and 'p1_damage_dealt_w1_rate' in df.columns:
        df['interact_p1_ratio_x_dmg_rate_w1'] = df['p1_team_offense_defense_ratio'] * df['p1_damage_dealt_w1_rate']
    if 'p2_team_offense_defense_ratio' in df.columns and 'p2_damage_dealt_w1_rate' in df.columns:
        df['interact_p2_ratio_x_dmg_rate_w1'] = df['p2_team_offense_defense_ratio'] * df['p2_damage_dealt_w1_rate']
    
    # HP × Accuracy: Capacità offensiva totale
    if 'p1_team_mean_base_hp' in df.columns and 'p1_avg_accuracy_used' in df.columns:
        df['v7_hp_x_accuracy'] = df['p1_team_mean_base_hp'] * df['p1_avg_accuracy_used']
    
    # Damage primo turno × Fainted: Early game pressure
    if 'p1_damage_dealt_w1' in df.columns and 'p1_fainted_pokemon' in df.columns:
        df['v7_damage_w1_x_fainted'] = df['p1_damage_dealt_w1'] * df['p1_fainted_pokemon']
    
    # Status × HP%: Effectiveness of status strategy
    if 'p1_status_inflicted' in df.columns and 'p1_avg_hp_pct_end' in df.columns:
        df['v7_status_x_hp'] = df['p1_status_inflicted'] * df['p1_avg_hp_pct_end']
    
    # Switch × Damage advantage: Tactical switching effectiveness
    if 'p1_switches' in df.columns and 'diff_damage_dealt' in df.columns:
        df['v7_switch_x_dmg_adv'] = df['p1_switches'] * df['diff_damage_dealt']
    
    # Type advantage × HP: Strategic type matchup
    if 'p1_super_effective_hits' in df.columns and 'p1_avg_hp_pct_end' in df.columns:
        df['v7_type_adv_x_hp'] = df['p1_super_effective_hits'] * df['p1_avg_hp_pct_end']
    
    # Fainted per turn: Eliminations efficiency
    if 'p1_fainted_pokemon_per_turn' in df.columns:
        df['v7_fainted_efficiency'] = df['p1_fainted_pokemon_per_turn']
    
    # Damage advantage × Accuracy: Offensive consistency
    if 'diff_damage_dealt' in df.columns and 'p1_avg_accuracy_used' in df.columns:
        df['v7_dmg_adv_x_acc'] = df['diff_damage_dealt'] * df['p1_avg_accuracy_used']
    
    # HP lost × Damage: Trading efficiency
    if 'p1_avg_hp_pct_start' in df.columns and 'p1_avg_hp_pct_end' in df.columns and 'p1_damage_dealt' in df.columns:
        hp_lost_pct = (df['p1_avg_hp_pct_start'] - df['p1_avg_hp_pct_end']) / 100
        df['v7_hp_lost_x_dmg'] = hp_lost_pct * df['p1_damage_dealt']
    
    # Super effective × HP (diff): Type advantage utilization
    if 'diff_super_effective_hits' in df.columns and 'diff_avg_hp_pct_end' in df.columns:
        df['v7_diff_se_x_hp'] = df['diff_super_effective_hits'] * df['diff_avg_hp_pct_end']
    
    # Switch × HP advantage: Tactical positioning
    if 'p1_switches' in df.columns and 'p1_avg_hp_pct_end' in df.columns and 'p2_avg_hp_pct_end' in df.columns:
        hp_adv = df['p1_avg_hp_pct_end'] - df['p2_avg_hp_pct_end']
        df['v7_switch_x_hp_adv'] = df['p1_switches'] * hp_adv
    
    # Momentum score: Overall offensive momentum
    if all(col in df.columns for col in ['diff_damage_dealt', 'p1_avg_accuracy_used', 'p1_avg_hp_pct_end']):
        df['v7_momentum'] = df['diff_damage_dealt'] * df['p1_avg_accuracy_used'] * (df['p1_avg_hp_pct_end'] / 100)
    
    # Offensive pressure: Sustained threat
    if all(col in df.columns for col in ['p1_damage_dealt', 'p1_super_effective_hits', 'p1_fainted_pokemon']):
        fainted_ratio = df['p1_fainted_pokemon'] / 6.0
        df['v7_offensive_pressure'] = df['p1_damage_dealt'] * df['p1_super_effective_hits'] * (1 + fainted_ratio)
    
    # Defensive stability: Maintain position
    if all(col in df.columns for col in ['p1_avg_hp_pct_start', 'p1_avg_hp_pct_end', 'diff_damage_dealt', 'p1_switches']):
        hp_lost_pct = (df['p1_avg_hp_pct_start'] - df['p1_avg_hp_pct_end']) / 100
        # Usa diff_damage_dealt come proxy per danno netto
        df['v7_defensive_stability'] = (1 - hp_lost_pct) * (1 + df['diff_damage_dealt'] / 100) * (1 + df['p1_switches'] / 10)
    
    # Tempo control: Early dominance
    if all(col in df.columns for col in ['p1_damage_dealt_w1', 'p1_damage_dealt_w2', 'p1_fainted_pokemon']):
        df['v7_tempo_control'] = (df['p1_damage_dealt_w1'] + df['p1_damage_dealt_w2']) * (df['p1_fainted_pokemon'] + 1) / 10
    
    # Endgame advantage: Close out ability
    if all(col in df.columns for col in ['p1_avg_hp_pct_end', 'p1_fainted_pokemon', 'diff_damage_dealt']):
        pokemon_alive = 6 - df['p1_fainted_pokemon']
        df['v7_endgame_advantage'] = df['p1_avg_hp_pct_end'] * pokemon_alive * (1 + df['diff_damage_dealt'] / 100)
    
    return df

def create_feature_df(file_path, max_turns=30):
    """Crea DataFrame di feature da un file JSONL di battaglie."""
    global TYPE_DISTRIBUTION_DF
    
    if TYPE_DISTRIBUTION_DF is None:
        try:
            TYPE_DISTRIBUTION_DF = load_type_distribution('predict.csv')
            print(f"✅ Distribuzione tipi caricata: {len(TYPE_DISTRIBUTION_DF)} tipi unici")
        except FileNotFoundError:
            print("⚠️  File predict.csv non trovato. Esegui prima predictor.py")
            print("   Le feature di predizione tipo saranno impostate a valori di default.")
    
    
    battles_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing battles"):
            battle_json = json.loads(line)
            battle_features = process_battle(battle_json, max_turns=max_turns)
            battles_data.append(battle_features)
    
    df = pd.DataFrame(battles_data)
    
    df = add_interaction_features(df)
    
    return df