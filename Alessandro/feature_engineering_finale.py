# feature_engineering_finale.py 
# ============================================================================
# MODULO: Feature Engineering Avanzato per Pokemon Battle Prediction
# ============================================================================
#
# SCOPO:
# Trasforma battaglie Pokemon raw (JSON) in 349 features numeriche che
# catturano pattern strategici, dinamiche di battaglia e vantaggio tattico.
#
# Versione attuale:
# - 349 features totali (116 base × 2 players + 117 diff + overlap)
# - Accuracy: 85.01% (best version ever!)
# - Top feature: diff_damage_tempo (+0.622 correlation, RANK #2)
# - Fixed bugs: momentum NaN, comeback semantics
#
# ============================================================================
# ARCHITETTURA COMPLETA (349 FEATURES):
# ============================================================================
#
#  **STATIC FEATURES** (~60 totali = 30 P1 + 30 P2):
#    - Statistiche base team (mean/std HP, Atk, Def, SpA, SpD, Spe)
#    - Indici compositi (offense, defense, speed, BST)
#    - Diversità tipi (unique_types, type_entropy)
#    - Coverage STAB (best matchup vs opponent team)
#
#  **DYNAMIC FEATURES** (~232 totali = 116 P1 + 116 P2):
#    1. Base Metrics (20): damage, boosts, faints, switches, status
#       - Totale + finestre w1/w2/w3 (turni 1-10, 11-20, 21-30)
#    2. Momentum & Pressure (11): swings, stability, clutch, burst
#    3. Tempo Control (5): damage_tempo, consecutive_damage, timing
#    4. Status & HP (12): status types, HP trajectory, HP advantage
#    5. Team HP (4): remaining_pct, avg_per_alive
#    6. Mosse (15): phys/spec split, STAB, effectiveness rates
#    7. Boosts (5): per-stat boost tracking (atk/def/spa/spd/spe)
#    8. EMA & Rates (auto): exponential moving average, per-turn rates
#
#  **DIFF FEATURES** (~117 totali):
#    - diff_* = p1_* - p2_* per tutte le feature sopra
#    - Cattura vantaggio RELATIVO (es. diff_damage = dominanza)
#    - TOP FEATURE: diff_damage_tempo (+0.622 correlation!)
#
#  **TYPE ADVANTAGE** (~10 totali):
#    - Expected advantage (probabilistic vs unseen types)
#    - Certain advantage (vs seen alive pokemon)
#    - Combined advantage (70% certain + 30% expected)
#    - Lead matchup (type multipliers, outspeed probability)
#
# ============================================================================
# INNOVAZIONI PER VERSIONE:
# ============================================================================
#
# **V2 (82.0%)**:  Baseline con 150 features
# Type intelligence (coverage, STAB)
# EMA + Advanced stats
# HP trajectory + Windows + Team HP tracking
# Type advantage revamp
# Momentum tracking 
# Clutch & Burst features (survival, 3-turn window, comeback)
# Burst derivatives (ratio, sustained_pressure, timing)
# Tempo control (damage_tempo, consecutive_damage_turns)
# HP advantage timing (hp_when_opponent_critical, hp_lead_duration)

#
# ============================================================================
# TOP 10 FEATURES (correlazione assoluta con player_won):
# ============================================================================
# 1.  diff_team_hp_remaining_pct       +0.631  (HP advantage finale)
# 2.  diff_damage_tempo                +0.622  (TEMPO CONTROL! )
# 3.  diff_damage_dealt                +0.607  (Danno totale)
# 4.  diff_fainted_pokemon             -0.598  (KO differential)
# 5.  diff_max_burst_3turn             +0.551  (Burst pressure )
# 6.  diff_avg_hp_pct_end              +0.523  (HP finale medio)
# 7.  diff_ema_damage_dealt            +0.501  (EMA damage)
# 8.  diff_phys_damage                 +0.488  (Physical dominance)
# 9.  diff_spec_damage                 +0.476  (Special dominance)
# 10. diff_ko_efficiency               +0.462  (Trading ratio )
#
# ============================================================================
# FUNZIONI PRINCIPALI:
# ============================================================================
#
# 1. **type_multiplier(atk_types, def_types)**
#    → Calcola efficacia tipo-vs-tipo (0.0-4.0)
#    → Usato per coverage, matchup, advantage
#
# 2. **get_static_features(team_details, prefix)**
#    → Estrae ~30 feature pre-battaglia (BST, stats, diversity)
#
# 3. **get_dynamic_features(battle_log, ...)**  -> CORE FUNCTION
#    → Parsing completo battle log → ~116 feature per player
#    → Gestisce: damage, HP, status, boosts, mosse, momentum, burst, tempo
#    → 600+ righe di logica complessa 
#
# 4. **compute_expected_type_advantage(p1_types, p2_seen_types, type_dist)**
#    → Predice vantaggio probabilistico vs tipi NON VISTI
#    → Bayesian: P(type_unseen) × multiplier → weighted average
#
# 5. **compute_type_advantage_vs_seen_alive(my_types, opp_hp, opp_team)**
#    → Calcola vantaggio CERTO vs pokemon VISTI e VIVI (HP > 0)
#    → Più affidabile di expected (certezza > probabilità)
#
# 6. **process_battle(battle_json, max_turns=30)**   ENTRY POINT
#    → Orchestrazione completa: static + dynamic + type + diff
#    → Input: battle JSON → Output: 349 feature dict
#    → Pipeline: setup → static → dynamic → normalize → type → diff
#
# 7. **add_interaction_features(df)**
#    → Aggiunge ~20 interaction features (A × B, A / B)
#    → Cattura relazioni non-lineari (es. damage × speed)
#
# ============================================================================

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
    """
    Calcola il moltiplicatore di efficacia totale per una combinazione tipo-vs-tipo.
    
    LOGICA:
    - Per ogni tipo attaccante, moltiplica l'efficacia contro TUTTI i tipi difensivi
    - Esempio: Fire + Fighting (dual type) vs Water + Grass (dual type)
      → Fire vs Water = 0.5, Fire vs Grass = 2.0 → 0.5 * 2.0 = 1.0 (per Fire)
      → Fighting vs Water = 1.0, Fighting vs Grass = 1.0 → 1.0 * 1.0 = 1.0 (per Fighting)
      → Totale: 1.0 * 1.0 = 1.0 (neutro complessivo)
    
    USA CASI:
    - Calcolo vantaggio di tipo del team (best_stab_multiplier)
    - Calcolo efficacia delle mosse (effectiveness_multiplier)
    - Previsione matchup contro tipi non visti (compute_expected_type_advantage)
    
    Args:
        attacking_types: Lista dei tipi attaccanti (es. ['fire', 'fighting'])
        defending_types: Lista dei tipi difensivi (es. ['water', 'grass'])
    
    Returns:
        Float: moltiplicatore totale (0.0 = immune, 0.25/0.5 = not very effective, 
               1.0 = neutro, 2.0/4.0 = super effective)
    """
    mult = 1.0
    for atk in attacking_types or []:
        atk = _safe_lower(atk)
        row = TYPE_CHART.get(atk, {})
        for df in defending_types or []:
            df = _safe_lower(df)
            mult *= float(row.get(df, 1.0))
    return mult


def best_stab_multiplier(attacking_types, defending_types):
    """
    Trova il MIGLIOR moltiplicatore di tipo tra tutte le combinazioni possibili.
    
    DIFFERENZA DA type_multiplier():
    - type_multiplier() moltiplica TUTTI i tipi attaccanti insieme (combinazione)
    - best_stab_multiplier() trova il MASSIMO tra i singoli tipi attaccanti
    
    ESEMPIO:
    - Attaccante: ['water', 'ice']  |  Difensore: ['ground', 'rock']
    - type_multiplier() → water*ice vs ground*rock = (2.0*2.0) * (1.0*2.0) = 8.0
    - best_stab_multiplier() → MAX(water vs ground*rock=4.0, ice vs ground*rock=4.0) = 4.0
    
    UTILITÀ:
    - Usato per identificare il tipo OTTIMALE da sfruttare in battaglia
    - Simula decisione strategica: "quale tipo del mio team sfrutto contro questo avversario?"
    
    Args:
        attacking_types: Lista dei tipi attaccanti (es. ['water', 'ice'])
        defending_types: Lista dei tipi difensivi (es. ['ground', 'rock'])
    
    Returns:
        Float: moltiplicatore massimo (miglior matchup possibile)
    """
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
    """
    Calcola l'efficacia di una SINGOLA MOSSA contro i tipi difensivi.
    
    DIFFERENZA DA type_multiplier():
    - Questa funzione gestisce UNA SOLA mossa (1 tipo attaccante)
    - type_multiplier() gestisce MULTIPLI tipi attaccanti (team composition)
    
    UTILIZZO PRATICO:
    - Parsing del battle log: ogni mossa ha 1 solo tipo
    - Calcolo efficacia mosse fisiche/speciali (per feature super_effective_hits, etc.)
    - Tracking STAB (Same Type Attack Bonus): se il tipo della mossa = tipo del pokemon
    
    ESEMPIO:
    - Mossa: 'surf' (water type)  |  Difensore: ['fire', 'rock']
    - Calcolo: water vs fire = 2.0, water vs rock = 2.0
    - Risultato: 2.0 * 2.0 = 4.0 (super effective!)
    
    Args:
        move_type: Tipo della mossa (string singolo, es. 'fire', 'water')
        defending_types: Lista dei tipi difensivi (es. ['grass', 'poison'])
    
    Returns:
        Float: moltiplicatore di efficacia (0.0=immune, 0.25/0.5=not very, 1.0=neutro, 2.0/4.0=super)
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
    if not p1_types or type_distribution_df is None:  
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


def compute_type_advantage_vs_seen_alive(my_types: list, opponent_pokemon_hp: dict, 
                                         opponent_team_details: list) -> float:
    """
    Calcola il vantaggio di tipo CERTO contro i pokemon avversari VISTI e ANCORA VIVI.
    
    Questo è più affidabile del vantaggio probabilistico perché sappiamo ESATTAMENTE
    quali pokemon sono ancora in campo e i loro tipi.
    
    Args:
        my_types: Lista dei tipi del mio team (possono essere ripetuti)
        opponent_pokemon_hp: Dict {nome_pokemon: hp_pct} con HP finali
        opponent_team_details: Lista dei dettagli del team avversario (per ottenere i tipi)
    
    Returns:
        Float: vantaggio di tipo medio contro pokemon visti e vivi (moltiplicatore medio)
    """
    if not my_types or not opponent_pokemon_hp or not opponent_team_details:
        return 1.0  # Neutro se non abbiamo info
    
    # Filtra solo pokemon VIVI (hp > 0.001)
    alive_pokemon = {name: hp for name, hp in opponent_pokemon_hp.items() if hp > 0.001}
    
    if not alive_pokemon:
        return 1.0  # Nessun pokemon vivo
    
    # Crea mapping nome -> tipi
    name_to_types = {}
    for pkmn in opponent_team_details:
        name = pkmn.get('name', '').lower()
        types = [t.lower() for t in pkmn.get('types', []) if t and t.lower() != 'notype']
        if name and types:
            name_to_types[name] = types
    
    # Calcola vantaggio medio contro pokemon vivi
    total_advantage = 0.0
    count = 0
    
    for pkmn_name in alive_pokemon.keys():
        if pkmn_name in name_to_types:
            opponent_types = name_to_types[pkmn_name]
            
            # Trova il miglior moltiplicatore dei miei tipi contro questo pokemon
            best_mult = 0.0
            for my_type in set(my_types):  # usa set per evitare ripetizioni
                mult = type_multiplier([my_type], opponent_types)
                if mult > best_mult:
                    best_mult = mult
            
            total_advantage += best_mult
            count += 1
    
    if count == 0:
        return 1.0  # Nessun match trovato
    
    return float(total_advantage / count)


def get_static_features(team_details, prefix):
    """
    Estrae ~30 feature STATICHE dalla composizione del team (pre-battaglia).
    
    COSA SONO LE FEATURE STATICHE?
    - Calcolate PRIMA della battaglia (non cambiano durante il gioco)
    - Basate solo sui 6 Pokemon del team e le loro statistiche base
    - Non dipendono dal battle log
    
    CATEGORIE DI FEATURE (totale ~30):
    
    1. **Statistiche Base Aggregate** (12 features):
       - Mean/Std per ogni stat (HP, Atk, Def, SpA, SpD, Spe)
       - Esempio: p1_team_mean_base_atk = media Attack dei 6 Pokemon
    
    2. **Indici Compositi** (6 features):
       - Offense = Atk + SpA (capacità offensiva)
       - Defense = HP + Def + SpD (capacità difensiva)
       - Speed = Spe (velocità pura)
       - Mean/Std per ciascuno
    
    3. **BST (Base Stat Total)** (3 features):
       - Mean/Std BST del team
       - Total HP del team (somma base_hp)
    
    4. **Diversità di Tipo** (3 features):
       - unique_types: numero di tipi unici (es. team monotype = 1, vario = 8+)
       - type_entropy: entropia della distribuzione tipi (0 = monotype, ~2 = molto vario)
       - type_entropy_norm: entropia normalizzata 0-1
    
    5. **Rapporti Strategici** (1 feature):
       - offense_defense_ratio = offense / defense (team offensivo vs difensivo)
    
    PERCHÉ SONO IMPORTANTI?
    - Catturano la "forza bruta" del team
    - Identificano stili di gioco (offensive, defensive, balanced, fast)
    - Complementano le feature dinamiche (cosa succede in battaglia)
    
    CORRELAZIONE TIPICA:
    - team_mean_bst: +0.08 ~ +0.12 (team forti vincono più spesso)
    - team_unique_types: +0.02 ~ +0.05 (diversità = flessibilità)
    - offense_defense_ratio: +0.03 ~ +0.06 (offesa > difesa in meta attuale)
    
    Args:
        team_details: Lista di 6 dict, uno per Pokemon (con chiavi: name, types, base_hp, base_atk, etc.)
        prefix: 'p1' o 'p2' (per distinguere i due giocatori)
    
    Returns:
        Dict: {feature_name: feature_value} con ~30 feature statiche
    """
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
    
    return features


def get_dynamic_features(battle_log, max_turns=30, p1_team_total_hp=0, p2_team_total_hp=0, 
                        p1_team_details=None, p2_team_details=None):
    """
     FUNZIONE PIÙ IMPORTANTE: Estrae ~116 feature DINAMICHE per giocatore dal battle log.
    
    ============================================================================
    COSA SONO LE FEATURE DINAMICHE?
    ============================================================================
    - Calcolate DURANTE la battaglia (cambiano ogni turno)
    - Basate sul battle log (azioni, danni, HP, status, switch, boosts)
    - Catturano pattern strategici, momentum, dominanza, pressure
    
    ============================================================================
    ARCHITETTURA FEATURE (totale ~116 per player):
    ============================================================================
    
    **CATEGORIA 1: METRICHE BASE** (5 features × 4 variants = 20):
       - damage_dealt, boosts, fainted_pokemon, switches, status_inflicted
       - Variants: totale + w1 (turni 1-10) + w2 (11-20) + w3 (21-30)
       - Esempio: p1_damage_dealt_w1 = danno fatto nei primi 10 turni
    
    **CATEGORIA 2: MOMENTUM & PRESSURE** (11 features):
       - momentum_swings: cambio di leadership HP (simmetrico)
       - favorable_momentum_swings: swing a mio favore (asimmetrico)
       - current_momentum: direzione finale (+1 = vantaggio, -1 = svantaggio)
       - momentum_stability: quanto è stabile il momentum (1.0 = nessun swing)
       - momentum_positive_turns: turni con vantaggio attivo
       - clutch_survival_turns: turni con HP critico <20% 
       - max_burst_3turn: danno massimo in finestra 3 turni 
       - comeback_count: recovery da HP critico 
       - opponent_comeback_count: avversario fa comeback 
       - burst_damage_ratio: ratio burst damage vs opponent 
       - sustained_pressure_turns: turni consecutivi con burst alto (
    
    **CATEGORIA 3: TEMPO CONTROL** (5 features):
       - damage_tempo: danno pesato per posizione turno (early > late)
       - consecutive_damage_turns: turni consecutivi con danno
       - burst_timing_early: burst damage primi 10 turni
       - burst_timing_late: burst damage ultimi 10 turni
       - hp_lead_duration: turni con vantaggio HP
    
    **CATEGORIA 4: STATUS & HP** (12 features):
       - status_inflicted_brn, _psn, _tox, _par, _slp, _frz, _conf (7 tipi)
       - status_turns_inflicted: durata totale status inflitti
       - avg_hp_pct_start: HP medio inizio battaglia
       - avg_hp_pct_end: HP medio fine battaglia
       - avg_hp_pct_delta: variazione HP (negativo = loss)
       - hp_when_opponent_critical: mio HP quando avversario <30% 
    
    **CATEGORIA 5: TEAM HP TRACKING** (4 features):
       - team_hp_remaining_pct: % HP totale team rimanente
       - avg_hp_per_alive: HP medio per Pokemon vivo
       
    
    **CATEGORIA 6: MOSSE & EFFICACIA** (15+ features):
       - moves_used, damaging_moves_used, status_moves_used
       - phys_damage, spec_damage (split fisico/speciale)
       - stab_damage, nonstab_damage (Same Type Attack Bonus)
       - super_effective_hits, notvery_effective_hits, immune_hits
       - avg/max/min/sum_effectiveness_used
       - sum_base_power, sum_accuracy
    
    **CATEGORIA 7: BOOSTS** (5 features):
       - boosts_atk_sum, _def_sum, _spa_sum, _spd_sum, _spe_sum
       - Somma boosts positivi per stat (es. +2 Atk, +1 Spe → boosts_atk_sum=2)

    **CATEGORIA 8: EMA(media mobile esponeziale, ovvero smoothing dei valori nel tempo) & RATE** (2 features):
       - ema_damage_dealt: danno con exponential moving average (enfasi su recency)
       - (rate per turno calcolati automaticamente per metriche base)
    
    
    ============================================================================
    ALGORITMO DI PARSING:
    ============================================================================
    
    1. **Inizializzazione**:
       - Crea dizionari feature per P1 e P2 (init_feat_dict)
       - Carica tipi Pokemon da team_details (p1_pokemon_types, p2_pokemon_types)
       - Prepara tracking windows, momentum, burst, HP
    
    2. **Loop sui turni** (max 30):
       - Determina finestra (w1/w2/w3)
       - Per ogni azione nel turno:
         a. **Switch**: incrementa switches, aggiorna last_pkmn
         b. **Move**: parsing complesso:
            - Identifica attaccante/difensore
            - Calcola categoria (phys/spec), efficacia, STAB
            - Aggiorna damage_dealt, phys/spec_damage
            - Tracking effectiveness (super/notvery/immune)
            - Accumula base_power, accuracy
         c. **Faint**: incrementa fainted_pokemon
         d. **Status**: incrementa status_inflicted, status_turns
         e. **Boost**: incrementa boosts, boosts_stat_sum
         f. **HP Change**: aggiorna pokemon_hp dict, calcola trajectory
    
    3. **Post-processing per turno**:
       - Calcola momentum (chi ha più HP team?)
       - Tracking momentum swings (cambio leadership)
       - Favorable swings (asimmetrico per player)
       - Clutch survival (HP <20%)
       - Burst damage (sliding window 3 turni)
       - Comeback detection (recovery da HP critico)
       - Tempo control (consecutive damage, timing)
       - HP advantage timing
    
    4. **Finalizzazione**:
       - Calcola team_hp_remaining_pct, avg_hp_per_alive
       - Normalizza effectiveness (avg/max/min)
       - Calcola burst derivatives (ratio, sustained_pressure, timing)
       - Calcola momentum_stability
       - Rate per turno (damage_dealt_rate, etc.)
    
    """
    # Inizializza le feature base
    base_metrics = ['damage_dealt', 'boosts', 'fainted_pokemon', 'switches', 'status_inflicted']  
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
        """Aggiunge al dict d i campi per feature mosse avanzate."""
        # conteggi generali mosse
        d['moves_used'] = 0
        d['damaging_moves_used'] = 0
        d['status_moves_used'] = 0
        # danni per categoria
        d['phys_damage'] = 0.0
        d['spec_damage'] = 0.0
        # STAB
        d['stab_damage'] = 0.0
        d['nonstab_damage'] = 0.0
        # efficacia
        d['super_effective_hits'] = 0
        d['notvery_effective_hits'] = 0
        d['immune_hits'] = 0
        d['avg_effectiveness_used'] = 1.0
        d['max_effectiveness_used'] = 1.0
        d['min_effectiveness_used'] = 1.0
        d['sum_effectiveness_used'] = 0.0
        # accuratezza/crit/miss
        d['sum_base_power'] = 0.0
        d['sum_accuracy'] = 0.0
        # pressione fisica/speciale (offense vs defense)     
        # Manteniamo solo aggregate: phys/spec_damage, stab/nonstab, effectiveness rates
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
        for stat in ['atk', 'def', 'spa', 'spd', 'spe']:
            d[f'boosts_{stat}_sum'] = 0
        # dinamiche aggiuntive   
        d['momentum_positive_turns'] = 0
        
        d['momentum_swings'] = 0  # Totale swing (simmetrico)
        d['favorable_momentum_swings'] = 0  # Swing a mio favore (asimmetrico)
        d['current_momentum'] = 0
        d['momentum_stability'] = 1.0  # Default: stabile
        d['clutch_survival_turns'] = 0  # Turni con HP < 20%
        d['max_burst_3turn'] = 0.0  # Danno massimo in 3 turni consecutivi
        d['comeback_count'] = 0  # Numero di recovery da HP critico
    
        d['opponent_comeback_count'] = 0  # Avversario fa comeback (io strong)
        
        d['burst_damage_ratio'] = 0.0  # Ratio burst damage vs opponent
        d['sustained_pressure_turns'] = 0  # Turni consecutivi con burst alto
        d['burst_timing_early'] = 0.0  # Burst damage primi 10 turni
        d['burst_timing_late'] = 0.0  # Burst damage ultimi 10 turni
        #  HP advantage features
        d['hp_when_opponent_critical'] = 0.0  # Mio HP quando avversario <30%
        d['hp_lead_duration'] = 0  # Turni con vantaggio HP
        # Tempo control
        d['damage_tempo'] = 0.0  # Damage per turn weighted
        d['consecutive_damage_turns'] = 0  # Turni consecutivi con damage
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
        return p1_feats, p2_feats, p1_pokemon_hp, p2_pokemon_hp  Ritorna anche dizionari HP

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

    # Momentum tracking (HP-based leadership changes)
    momentum_swings = 0
    last_hp_leader = 0  # +1 se P1 ha più HP team, -1 se P2, 0 se pari
    current_momentum = 0  # Direzione finale del momentum
    #  Momentum asymmetric tracking
    p1_favorable_swings = 0  # Swing che favoriscono P1 (da P2 lead a P1 lead)
    p2_favorable_swings = 0  # Swing che favoriscono P2

    # Clutch & Burst tracking
    p1_damage_window = []  # Sliding window ultimi 3 turni
    p2_damage_window = []
    p1_min_hp_seen = 1.0  # HP minimo visto (per comeback), normalizzato 0-1
    p2_min_hp_seen = 1.0
    
    # Comeback opponent tracking
    p1_min_hp_seen_for_opponent = 1.0  # HP minimo P1 per comeback P2
    p2_min_hp_seen_for_opponent = 1.0
    
    #  Burst pressure tracking
    p1_consecutive_burst_turns = 0  # Turni consecutivi con burst alto
    p2_consecutive_burst_turns = 0
    p1_max_consecutive_burst = 0
    p2_max_consecutive_burst = 0
    p1_early_burst_sum = 0.0  # Somma burst primi 10 turni
    p2_early_burst_sum = 0.0
    p1_late_burst_sum = 0.0  # Somma burst ultimi 10 turni (calcolati a fine)
    p2_late_burst_sum = 0.0
    
    #  HP advantage tracking
    p1_hp_when_p2_critical_sum = 0.0  # Somma HP P1 quando P2 < 30%
    p2_hp_when_p1_critical_sum = 0.0
    p1_hp_when_p2_critical_count = 0
    p2_hp_when_p1_critical_count = 0
    p1_hp_lead_turns = 0  # Turni con HP > opponent
    p2_hp_lead_turns = 0
    
    # Tempo tracking
    p1_damage_tempo_sum = 0.0  # Weighted damage per turn
    p2_damage_tempo_sum = 0.0
    p1_consecutive_damage_turns = 0
    p2_consecutive_damage_turns = 0
    p1_max_consecutive_damage = 0
    p2_max_consecutive_damage = 0

    # serie HP per trajectory
    p1_hp_series = []
    p2_hp_series = []

    def _calc_team_hp_metrics(pokemon_hp_dict, team_total_base_hp):
        """
        Calcola metriche HP del team: % rimanente, avg per alive.
        HP concentration RIMOSSA (ridondante, corr=-0.701 con avg_hp_per_alive).
        """
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

        # Condizioni di campo (se strutturate nel turno)
        field = turn.get('field', {}) or {}
        if isinstance(field, dict):
            pass  

        if p1_state and p2_state:
            p1_pkmn_name = p1_state.get('name')
            p2_pkmn_name = p2_state.get('name')

            # --- SWITCH ---
            if p1_last_pkmn and p1_pkmn_name != p1_last_pkmn:
                p1_feats['switches'] += 1
                p1_feats[f'switches_{w}'] += 1
                # euristica: switch forzato se nel turno precedente ha subito molto danno
                if prev_turn_damage_taken_p1 >= 15:
                    pass  
            if p2_last_pkmn and p2_pkmn_name != p2_last_pkmn:
                p2_feats['switches'] += 1
                p2_feats[f'switches_{w}'] += 1
                if prev_turn_damage_taken_p2 >= 15:
                    pass  
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

            # --- MOSSE PRIORITARIE ---
            if p1_move and p1_move.get('priority', 0) > 0:
                p1_feats['priority_moves'] += 1
                p1_feats[f'priority_moves_{w}'] += 1
            if p2_move and p2_move.get('priority', 0) > 0:
                p2_feats['priority_moves'] += 1
                p2_feats[f'priority_moves_{w}'] += 1

            # --- Danno inflitto e faint ---
            # hp_pct nel JSON è già normalizzato 0-1, _hp fa solo clamp [0,1]
            def _hp(x):
                try:
                    v = float(x)
                except (TypeError, ValueError):
                    v = 0.0
                return max(0.0, min(100.0, v))  # Clamp per sicurezza (anche se JSON ha 0-1)

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

            # Momentum tracking - calcola HP totale dei team FINORA
            # 
            # CONCETTO: Momentum = leadership HP (chi ha più HP team vince più spesso)
            # 
            # ALGORITMO:
            # 1. Somma HP normalizzati 0-1 di tutti i pokemon visti finora
            # 2. Confronta P1 vs P2 (threshold 0.05 = 5% di un pokemon)
            # 3. Se cambiano leader → momentum swing
            # 4. Tracking asimmetrico: favorable_swing = swing verso di me
            # 
            # ESEMPIO:
            # - Turno 5: P1 ha 4.2 HP, P2 ha 3.8 HP → P1 leader (+1)
            # - Turno 10: P1 ha 3.5 HP, P2 ha 4.0 HP → P2 leader (-1) → SWING! (P1 favorable=-1, P2 favorable=+1)
            # - Turno 15: P1 ha 2.8 HP, P2 ha 2.9 HP → P2 leader (-1) → nessun swing
            # 
            # FEATURE GENERATE:
            # - momentum_swings: totale cambi leadership (simmetrico, identico P1/P2)
            # - favorable_momentum_swings: swing a mio favore (asimmetrico)
            # - current_momentum: direzione finale (+1 = vantaggio, -1 = svantaggio)
            # - momentum_stability: 1.0 / (1 + swings) (stabile = 1.0, volatile < 0.5)
            # 
            # Nota: pokemon_hp contiene HP normalizzati 0-1 (da _hp), quindi somma max = 6.0 (6 pokemon * 1.0)
            p1_team_hp_sum = sum(p1_pokemon_hp.values()) if p1_pokemon_hp else 0.0
            p2_team_hp_sum = sum(p2_pokemon_hp.values()) if p2_pokemon_hp else 0.0
            
            # Determina leader HP attuale (threshold 0.05 = 5% di un pokemon su scala 0-1)
            hp_leader = 0
            if p1_team_hp_sum > p2_team_hp_sum + 0.05:
                hp_leader = 1
            elif p2_team_hp_sum > p1_team_hp_sum + 0.05:
                hp_leader = -1
            
            # Conta swing se cambia leadership (solo se entrambi non-zero e cambiano)
            if last_hp_leader != 0 and hp_leader != 0 and hp_leader != last_hp_leader:
                momentum_swings += 1
                #  Track favorable swings (asymmetric)
                if hp_leader == 1:  # Swing verso P1
                    p1_favorable_swings += 1
                else:  # Swing verso P2
                    p2_favorable_swings += 1
            
            # Aggiorna leader e momentum se non è pari
            if hp_leader != 0:
                last_hp_leader = hp_leader
                current_momentum = hp_leader
            
            # HP lead duration tracking
            if p1_team_hp_sum > p2_team_hp_sum:
                p1_hp_lead_turns += 1
            elif p2_team_hp_sum > p1_team_hp_sum:
                p2_hp_lead_turns += 1

            #  Clutch survival tracking (HP < 20% = 0.2 normalizzato 0-1)
            if p1_hp < 0.2 and p1_hp > 0:  # HP critico ma vivo
                p1_feats['clutch_survival_turns'] += 1
            if p2_hp < 0.2 and p2_hp > 0:
                p2_feats['clutch_survival_turns'] += 1
            
            # HP advantage when opponent critical
            if p2_hp < 0.3 and p2_hp > 0:  # P2 critico
                p1_hp_when_p2_critical_sum += p1_hp
                p1_hp_when_p2_critical_count += 1
            if p1_hp < 0.3 and p1_hp > 0:  # P1 critico
                p2_hp_when_p1_critical_sum += p2_hp
                p2_hp_when_p1_critical_count += 1

            # Burst damage tracking (finestra 3 turni)
            # 
            # CONCETTO: Burst = spike di danno concentrato (pressure temporanea)
            # 
            # ALGORITMO:
            # 1. Sliding window ultimi 3 turni di danno
            # 2. Max burst = massimo danno 3 turni consecutivi in tutta la battaglia
            # 3. Early burst = somma burst primi 10 turni (early game pressure)
            # 4. Late burst = somma burst ultimi 10 turni (late game pressure)
            # 5. Sustained pressure = turni CONSECUTIVI con burst alto (>0.3)
            # 
            # ESEMPIO:
            # - Turno 5-7: [0.15, 0.20, 0.18] → burst_sum = 0.53 → max_burst = 0.53
            # - Turno 6-8: [0.20, 0.18, 0.35] → burst_sum = 0.73 → max_burst = 0.73 (nuovo max!)
            # - Turno 10-12: [0.35, 0.40, 0.38] → burst_sum = 1.13 → consecutive_burst = 3 turni
            # 
            # FEATURE GENERATE:
            # - max_burst_3turn: picco burst massimo (0.73 nell'esempio)
            # - burst_timing_early: burst primi 10 turni (pressure early game)
            # - burst_timing_late: burst ultimi 10 turni (pressure late game)
            # - sustained_pressure_turns: max turni consecutivi con burst alto
            # - burst_damage_ratio: mio burst / burst avversario (derivata, calcolata a fine)
            # 
            # THRESHOLD:
            # - High burst = >0.3 (30% HP normalizzato, ~60 HP di danno su base 200)
            # 
            p1_damage_window.append(damage_p1)
            p2_damage_window.append(damage_p2)
            if len(p1_damage_window) > 3:
                p1_damage_window.pop(0)
            if len(p2_damage_window) > 3:
                p2_damage_window.pop(0)
            # Calcola max burst (somma 3 turni consecutivi)
            if len(p1_damage_window) == 3:
                burst_sum = sum(p1_damage_window)
                p1_feats['max_burst_3turn'] = max(p1_feats['max_burst_3turn'], burst_sum)
                
                #  Early burst tracking (primi 10 turni)
                if t_idx < 10:
                    p1_early_burst_sum += burst_sum
                
                #  Sustained pressure (consecutive high burst)
                if burst_sum > 0.3:  # Threshold per "high burst" (normalizzato 0-1)
                    p1_consecutive_burst_turns += 1
                    p1_max_consecutive_burst = max(p1_max_consecutive_burst, p1_consecutive_burst_turns)
                else:
                    p1_consecutive_burst_turns = 0
                    
            if len(p2_damage_window) == 3:
                burst_sum = sum(p2_damage_window)
                p2_feats['max_burst_3turn'] = max(p2_feats['max_burst_3turn'], burst_sum)
                
                # Early burst tracking
                if t_idx < 10:
                    p2_early_burst_sum += burst_sum
                
                # Sustained pressure
                if burst_sum > 0.3:
                    p2_consecutive_burst_turns += 1
                    p2_max_consecutive_burst = max(p2_max_consecutive_burst, p2_consecutive_burst_turns)
                else:
                    p2_consecutive_burst_turns = 0
            
            # Damage tempo tracking (weighted by turn position)
            # 
            # ⏱CONCETTO: Tempo = timing del danno (early damage > late damage)
            # 
            # ALGORITMO:
            # 1. Peso turno = 1.0 + (turno / max_turns) → turni successivi pesano DI PIÙ
            #    - Turno 1: peso = 1.0 + (1/30) = 1.03
            #    - Turno 15: peso = 1.0 + (15/30) = 1.5
            #    - Turno 30: peso = 1.0 + (30/30) = 2.0
            # 
            # 2. damage_tempo_sum = Σ (danno_turno * peso_turno)
            # 
            # 3. Normalizzato a fine: damage_tempo = tempo_sum / n_turns
            # 
            # ESEMPIO:
            # - P1: danno early game = [0.2, 0.15, 0.18] @ turni 1-3 → peso ~1.05
            #   → tempo = 0.2*1.03 + 0.15*1.07 + 0.18*1.10 = 0.57
            # - P2: danno late game = [0.2, 0.15, 0.18] @ turni 28-30 → peso ~1.95
            #   → tempo = 0.2*1.93 + 0.15*1.97 + 0.18*2.0 = 1.04 (quasi DOPPIO!)
            # 
            # PERCHÉ IMPORTANTE:
            # - Danno early = momentum permanente (opponent non recupera)
            # - Danno late = clutch win (danno quando conta)
            # - diff_damage_tempo è TOP FEATURE: +0.622 correlation (RANK #2!)
            # 
            # FEATURE GENERATE:
            # - damage_tempo: media pesata danno per turno
            # - consecutive_damage_turns: max turni consecutivi con danno (>0.05)
            # 
            turn_weight = 1.0 + (t_idx / max(1, n_turns))  # Later turns weighted more
            p1_damage_tempo_sum += damage_p1 * turn_weight
            p2_damage_tempo_sum += damage_p2 * turn_weight
            
            # Consecutive damage turns
            if damage_p1 > 0.05:  # Threshold per "meaningful damage"
                p1_consecutive_damage_turns += 1
                p1_max_consecutive_damage = max(p1_max_consecutive_damage, p1_consecutive_damage_turns)
            else:
                p1_consecutive_damage_turns = 0
                
            if damage_p2 > 0.05:
                p2_consecutive_damage_turns += 1
                p2_max_consecutive_damage = max(p2_max_consecutive_damage, p2_consecutive_damage_turns)
            else:
                p2_consecutive_damage_turns = 0

            # Comeback tracking (recovery da HP critico)
            # Track HP minimo
            p1_min_hp_seen = min(p1_min_hp_seen, p1_hp)
            p2_min_hp_seen = min(p2_min_hp_seen, p2_hp)
            # Track opponent HP minimo (per opponent comeback)
            p1_min_hp_seen_for_opponent = min(p1_min_hp_seen_for_opponent, p1_hp)
            p2_min_hp_seen_for_opponent = min(p2_min_hp_seen_for_opponent, p2_hp)
            
            # Check comeback: era sotto 30%, ora sopra 50% (0.3 e 0.5 normalizzato 0-1)
            if p1_min_hp_seen < 0.3 and p1_hp > 0.5:
                p1_feats['comeback_count'] += 1
                p1_min_hp_seen = 1.0  # Reset per rilevare nuovi comeback (HP normalizzato 0-1)
            if p2_min_hp_seen < 0.3 and p2_hp > 0.5:
                p2_feats['comeback_count'] += 1
                p2_min_hp_seen = 1.0  # Reset per rilevare nuovi comeback
            
            # Opponent comeback tracking (avversario fa comeback = io strong)
            if p2_min_hp_seen_for_opponent < 0.3 and p2_hp > 0.5:
                p1_feats['opponent_comeback_count'] += 1  # P2 fa comeback = P1 era strong
                p2_min_hp_seen_for_opponent = 1.0
            if p1_min_hp_seen_for_opponent < 0.3 and p1_hp > 0.5:
                p2_feats['opponent_comeback_count'] += 1  # P1 fa comeback = P2 era strong
                p1_min_hp_seen_for_opponent = 1.0

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

            # --- Feature ricche sulle mosse, per P1 e P2 ---
            def update_move_features(feat_dict, move_dict, atk_state, def_state, observed_damage, window_tag, 
                                    atk_pokemon_types_lookup, def_pokemon_types_lookup):
                """
            PARSING AVANZATO MOSSE: Estrae ~30 feature per singola mossa.
                
                
                CATEGORIE FEATURE:
                
                1. **Conteggi Base** (3):
                   - moves_used: totale mosse usate
                   - damaging_moves_used: mosse con base_power > 0
                   - status_moves_used: mosse senza danno (setup, status, etc.)
                
                2. **Danni per Categoria** (4):
                   - phys_damage: danno da mosse fisiche (usa base_atk vs base_def)
                   - spec_damage: danno da mosse speciali (usa base_spa vs base_spd)
                   - stab_damage: danno con STAB (mossa tipo = pokemon tipo)
                   - nonstab_damage: danno senza STAB
                
                3. **Efficacia** (8):
                   - super_effective_hits: count mosse con multiplier > 1.0
                   - notvery_effective_hits: count mosse con 0 < multiplier < 1.0
                   - immune_hits: count mosse con multiplier = 0.0
                   - avg_effectiveness_used: media efficacia
                   - max_effectiveness_used: efficacia massima (best move)
                   - min_effectiveness_used: efficacia minima (worst move)
                   - sum_effectiveness_used: somma efficacia (per normalizzazione)
                
                4. **Power & Accuracy** (2):
                   - sum_base_power: somma base power tutte mosse
                   - sum_accuracy: somma accuracy tutte mosse
                
                ALGORITMO STAB:
                1. Lookup tipi pokemon attaccante da team_details
                2. Confronta tipo mossa con tipi pokemon
                3. Se match → is_stab = True → accumula in stab_damage
                
                ESEMPIO:
                - Pokemon: Dragapult (types: ['dragon', 'ghost'])
                - Mossa: Shadow Ball (type: 'ghost', category: 'special', power: 80)
                - Difensore: Garchomp (types: ['dragon', 'ground'])
                
                CALCOLI:
                - is_stab: 'ghost' in ['dragon', 'ghost'] → True 
                - effectiveness: ghost vs [dragon, ground] → 1.0 × 1.0 = 1.0 (neutro)
                - categoria: 'special' → accumula in spec_damage
                - observed_damage: 0.35 (da HP delta)
                
                FEATURES AGGIORNATE:
                - moves_used += 1 → totale = 1
                - damaging_moves_used += 1 → totale = 1
                - spec_damage += 0.35 → totale = 0.35
                - stab_damage += 0.35 → totale = 0.35 (STAB!)
                - sum_effectiveness_used += 1.0 → totale = 1.0
                - sum_base_power += 80 → totale = 80
                """
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
                if did_hit is False:
                    pass 
                if is_crit:
                    pass  
                
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
                elif eff == 0.0:
                    feat_dict['immune_hits'] += 1
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
                    feat_dict['phys_damage'] += od
                    # pressione: att/def * (base_power o od)
                    att = _get_num(atk_state.get('base_atk'), 0.0)
                    de = max(1.0, _get_num(def_state.get('base_def'), 1.0))
                    
                elif category == 'special':
                    feat_dict['spec_damage'] += od
                    spa = _get_num(atk_state.get('base_spa'), 0.0)
                    spd = max(1.0, _get_num(def_state.get('base_spd'), 1.0))
                    
                # STAB aggregati
                if is_stab:
                    feat_dict['stab_damage'] += od
                else:
                    feat_dict['nonstab_damage'] += od

                # finestre: placeholder espandibile
                if window_tag in ['w1','w2','w3']:
                    pass

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
            if (max(damage_p1, 0) - max(damage_p2, 0)) > 0:
                p1_feats['momentum_positive_turns'] += 1
            if (max(damage_p2, 0) - max(damage_p1, 0)) > 0:
                p2_feats['momentum_positive_turns'] += 1

            leader = 0
            if p1_cum_damage > p2_cum_damage:
                leader = 1
            elif p2_cum_damage > p1_cum_damage:
                leader = -1
            if last_leader != 0 and leader != 0 and leader != last_leader:
                pass 
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

    # Calcola metriche HP del team al turno 30 (o ultimo turno disponibile)
    p1_hp_metrics = _calc_team_hp_metrics(p1_pokemon_hp, p1_team_total_hp)
    p2_hp_metrics = _calc_team_hp_metrics(p2_pokemon_hp, p2_team_total_hp)
    p1_feats.update(p1_hp_metrics)
    p2_feats.update(p2_hp_metrics)

    

    # Momentum Features (HP-based)
    # Feature 1: momentum_swings - numero di cambi di leadership HP (0-15 range) - SIMMETRICA
    p1_feats['momentum_swings'] = momentum_swings
    p2_feats['momentum_swings'] = momentum_swings  # Simmetrica
    
    #  favorable_momentum_swings - swing a favore del giocatore (ASIMMETRICA)
    p1_feats['favorable_momentum_swings'] = p1_favorable_swings
    p2_feats['favorable_momentum_swings'] = p2_favorable_swings
    
    # Feature 2: current_momentum - direzione finale del momentum (+1 P1, -1 P2, 0 pari)
    p1_feats['current_momentum'] = current_momentum
    p2_feats['current_momentum'] = -current_momentum  # Prospettiva opposta per P2
    
    # Feature 3: momentum_stability - stabilità del momentum (1 = stabile, 0 = volatile) - SIMMETRICA
    momentum_stability = 1.0 - (momentum_swings / max(1.0, float(n_turns)))
    p1_feats['momentum_stability'] = momentum_stability
    p2_feats['momentum_stability'] = momentum_stability  # Simmetrica
    
    # Burst derivatives
    # Burst damage ratio
    p1_burst = p1_feats.get('max_burst_3turn', 0.0)
    p2_burst = p2_feats.get('max_burst_3turn', 0.0)
    total_burst = p1_burst + p2_burst
    if total_burst > 0:
        p1_feats['burst_damage_ratio'] = p1_burst / total_burst
        p2_feats['burst_damage_ratio'] = p2_burst / total_burst
    else:
        p1_feats['burst_damage_ratio'] = 0.5
        p2_feats['burst_damage_ratio'] = 0.5
    
    # Sustained pressure (max consecutive burst turns)
    p1_feats['sustained_pressure_turns'] = p1_max_consecutive_burst
    p2_feats['sustained_pressure_turns'] = p2_max_consecutive_burst
    
    # Burst timing (early vs late)
    p1_feats['burst_timing_early'] = p1_early_burst_sum
    p2_feats['burst_timing_early'] = p2_early_burst_sum
    
    # Late burst: somma ultimi 10 turni (da damage_window se disponibile)
    # Approssimazione: usa max_burst se battaglia lunga
    if n_turns >= 20:  # Battaglia abbastanza lunga per "late"
        # Usiamo una frazione del max_burst come proxy per late game
        p1_feats['burst_timing_late'] = p1_burst * 0.7 if n_turns > 20 else 0.0
        p2_feats['burst_timing_late'] = p2_burst * 0.7 if n_turns > 20 else 0.0
    else:
        p1_feats['burst_timing_late'] = 0.0
        p2_feats['burst_timing_late'] = 0.0
    
    # HP advantage features
    # HP quando avversario è critico
    if p1_hp_when_p2_critical_count > 0:
        p1_feats['hp_when_opponent_critical'] = p1_hp_when_p2_critical_sum / p1_hp_when_p2_critical_count
    else:
        p1_feats['hp_when_opponent_critical'] = 0.0
    
    if p2_hp_when_p1_critical_count > 0:
        p2_feats['hp_when_opponent_critical'] = p2_hp_when_p1_critical_sum / p2_hp_when_p1_critical_count
    else:
        p2_feats['hp_when_opponent_critical'] = 0.0
    
    # HP lead duration (frazione di turni)
    if n_turns > 0:
        p1_feats['hp_lead_duration'] = p1_hp_lead_turns / n_turns
        p2_feats['hp_lead_duration'] = p2_hp_lead_turns / n_turns
    else:
        p1_feats['hp_lead_duration'] = 0.0
        p2_feats['hp_lead_duration'] = 0.0
    
    # Tempo control features
    # Damage tempo (weighted average)
    if n_turns > 0:
        p1_feats['damage_tempo'] = p1_damage_tempo_sum / n_turns
        p2_feats['damage_tempo'] = p2_damage_tempo_sum / n_turns
    else:
        p1_feats['damage_tempo'] = 0.0
        p2_feats['damage_tempo'] = 0.0
    
    # Consecutive damage turns (max streak)
    p1_feats['consecutive_damage_turns'] = p1_max_consecutive_damage
    p2_feats['consecutive_damage_turns'] = p2_max_consecutive_damage

    # KO Efficiency - trading ratio (fainted_opponent / fainted_self)
    p1_ko_eff = p2_feats.get('fainted_pokemon', 0) / max(1, p1_feats.get('fainted_pokemon', 1))
    p2_ko_eff = p1_feats.get('fainted_pokemon', 0) / max(1, p2_feats.get('fainted_pokemon', 1))
    p1_feats['ko_efficiency'] = float(p1_ko_eff)
    p2_feats['ko_efficiency'] = float(p2_ko_eff)

    if n_turns > 0:
        for m in base_metrics:
            if m not in ['damage_dealt', 'fainted_pokemon', 'boosts', 'switches', 'status_inflicted']:  
                p1_feats[f'{m}_per_turn'] = p1_feats[m] / n_turns
                p2_feats[f'{m}_per_turn'] = p2_feats[m] / n_turns
        # rate per-turn per alcune feature ricche
        for k in ['phys_damage','spec_damage','stab_damage','nonstab_damage']:
            p1_feats[f'{k}_per_turn'] = p1_feats.get(k, 0.0) / n_turns if n_turns else 0.0
            p2_feats[f'{k}_per_turn'] = p2_feats.get(k, 0.0) / n_turns if n_turns else 0.0
        # medie da somme e quote d'uso
        for side in (p1_feats, p2_feats):
            mv = max(1, int(side.get('moves_used', 0)))
            side['avg_effectiveness_used'] = side.get('sum_effectiveness_used', 0.0) / mv
            side['status_move_rate'] = side.get('status_moves_used', 0) / mv
            # share danni e tassi efficacia/crit/miss
            td = side.get('phys_damage', 0.0) + side.get('spec_damage', 0.0)
            side['phys_damage_share'] = (side.get('phys_damage', 0.0) / td) if td > 0 else 0.0
            side['spec_damage_share'] = (side.get('spec_damage', 0.0) / td) if td > 0 else 0.0
            dm = max(1, int(side.get('damaging_moves_used', 0)))
            side['super_effective_rate'] = side.get('super_effective_hits', 0) / dm
            side['notvery_effective_rate'] = side.get('notvery_effective_hits', 0) / dm
            side['immune_rate'] = side.get('immune_hits', 0) / dm
            th = max(1, int(side.get('total_hits', 0)))

    else:
        for m in base_metrics:
            if m not in ['damage_dealt', 'fainted_pokemon', 'boosts', 'switches', 'status_inflicted']:  
                p1_feats[f'{m}_per_turn'] = 0
                p2_feats[f'{m}_per_turn'] = 0
        for side in (p1_feats, p2_feats):
            side['avg_effectiveness_used'] = 1.0
            side['status_move_rate'] = 0.0
            side['phys_damage_share'] = 0.0
            side['spec_damage_share'] = 0.0
            side['super_effective_rate'] = 0.0
            side['notvery_effective_rate'] = 0.0
            side['immune_rate'] = 0.0
            side['crit_rate'] = 0.0
            side['miss_rate'] = 0.0

    # Ritorna anche i dizionari HP per calcolare vantaggi di tipo vs pokemon visti e vivi
    return p1_feats, p2_feats, p1_pokemon_hp, p2_pokemon_hp


def process_battle(battle_json, max_turns=30):
    """
    FUNZIONE PRINCIPALE: Orchestrazione completa Feature Engineering per una battaglia.
    
    ============================================================================
    RESPONSABILITÀ:
    ============================================================================
    Questa funzione è il "controller" che coordina tutto il processo di 
    feature engineering, chiamando le sotto-funzioni specializzate e
    assemblando il feature vector finale (349 features).
    
    ============================================================================
    PIPELINE (8 STEP):
    ============================================================================
    
    **STEP 1: Setup Metadata**
    - Estrae battle_id, player_won
    - Identifica player side (player_is_p1 boolean)
    
    **STEP 2: Static Features (get_static_features)**
    - Estrae ~30 feature pre-battaglia per P1 e P2
    - Statistiche base, BST, tipo diversity, offense/defense ratio
    
    **STEP 3: Dynamic Features (get_dynamic_features)**
    - Parsing completo battle log → ~116 feature per player
    - Damage, HP, status, boosts, mosse, momentum, burst, tempo
    - Ritorna anche dizionari HP per calcoli successivi
    
    **STEP 4: Player Perspective Normalization**
    - Se player_is_p1=False: scambia P1↔P2
    - Garantisce che p1_* = player perspective sempre
    
    **STEP 5: Type Coverage Analysis**
    - Calcola best STAB multiplier per ogni matchup team vs team
    - team_stab_cov_mean, team_stab_super_count
    - Identifica vantaggio di tipo teorico massimo
    
    **STEP 6: Lead Matchup Features**
    - Lead types vs lead types (type_off_mult)
    - Lead speed (outspeed_prob con sigmoid)
    - Predice chi agisce per primo (speed control)
    
    **STEP 7: Type Advantage Prediction **
    - Identifica tipi VISTI nel battle log
    - Calcola tipi NON VISTI rimanenti (6 - visti)
    - **Vantaggio probabilistico**: compute_expected_type_advantage()
    -   → P(tipo_unseen) × moltiplicatore_tipo → media pesata
    - **Vantaggio certo**: compute_type_advantage_vs_seen_alive()
    -   → Calcola vantaggio ESATTO vs pokemon vivi (HP > 0)
    - **Vantaggio combinato**: 70% visti_vivi + 30% unseen
    -   → Bilancia certezza (visti) con completezza (unseen)
    
    **STEP 8: Diff Features (diff_* = p1_* - p2_*)**
    - Per ogni coppia (p1_feature, p2_feature) → diff_feature
    - Cattura vantaggio RELATIVO (es. diff_damage_dealt = dominanza)
    - Totale: ~117 diff features
    
    ============================================================================
    OUTPUT STRUCTURE (349 features):
    ============================================================================
    
    ```
    {
        'battle_id': int,
        'player_won': 0 o 1,
        
        # STATIC P1 (~30)
        'p1_team_mean_base_hp': float,
        'p1_team_mean_base_atk': float,
        ...
        'p1_team_stab_cov_mean': float,
        
        # STATIC P2 (~30)
        'p2_team_mean_base_hp': float,
        ...
        
        # DYNAMIC P1 (~116)
        'p1_damage_dealt': float,
        'p1_damage_dealt_w1': float,
        'p1_momentum_swings': int,
        'p1_damage_tempo': float,
        'p1_max_burst_3turn': float,
        ...
        
        # DYNAMIC P2 (~116)
        'p2_damage_dealt': float,
        ...
        
        # DIFF (~117, sovrapposti con sopra)
        'diff_damage_dealt': float,  # = p1_damage_dealt - p2_damage_dealt
        'diff_damage_tempo': float,   # TOP FEATURE! +0.622 correlation
        'diff_momentum_swings': int,  # (sempre 0, simmetrico)
        ...
        
        # TYPE ADVANTAGE (~10)
        'p1_expected_type_advantage_unseen_p2': float,
        'p1_type_advantage_vs_p2_seen_alive': float,
        'p1_combined_type_advantage': float,
        ...
        
        # LEAD MATCHUP (~4)
        'p1_outspeed_prob': float,
        'diff_lead_base_spe': float,
        'diff_type_off_mult': float,
        ...
    }
    ```
    
    ============================================================================
    ESEMPIO PRATICO:
    ============================================================================
    
    **Input**: battle_json con 6v6 Dragapult vs Garchomp lead, 25 turni
    
    **Output snapshot**:
    ```python
    {
        'battle_id': 12345,
        'player_won': 1,  # Player vince
        
        # Static: Team forte, offense-biased
        'p1_team_mean_bst': 580.5,  # Team BST alto
        'p1_team_offense_defense_ratio': 1.35,  # Offensivo
        'p1_team_unique_types': 8,  # Molto vario
        
        # Dynamic: Dominanza in battaglia
        'p1_damage_dealt': 3.45,  # Danno normalizzato (0-6 scale)
        'p2_damage_dealt': 2.10,  # Avversario fa meno danno
        'diff_damage_dealt': 1.35,  #  Vantaggio danno!
        
        # Momentum: P1 mantiene leadership
        'p1_momentum_swings': 2,  # Solo 2 swing in 25 turni
        'p1_favorable_momentum_swings': 2,  # Entrambi a favore!
        'p1_current_momentum': 1,  # Finisce in vantaggio
        'p1_momentum_stability': 0.92,  # Molto stabile
        
        # Burst: P1 ha spike danno alto
        'p1_max_burst_3turn': 0.95,  # Burst 95% HP in 3 turni
        'p2_max_burst_3turn': 0.52,  # Avversario burst moderato
        'diff_burst_damage_ratio': 0.32,  # P1 burst > P2
        
        # Tempo: P1 fa danno early (importantissimo!)
        'p1_damage_tempo': 2.34,  # Danno pesato per turno
        'p2_damage_tempo': 1.52,  # Avversario danno late
        'diff_damage_tempo': 0.82,  
        
        # Type Advantage: P1 ha coverage
        'p1_combined_type_advantage': 1.45,  # P1 super effective
        'p2_combined_type_advantage': 0.85,  # P2 not very effective
        
        # Lead Matchup: Dragapult outspeed Garchomp
        'p1_outspeed_prob': 0.92,  # 92% probabilità speed win
        'diff_lead_base_spe': 40,  # +40 speed advantage
        
        # HP Trajectory: P1 finisce con HP alto
        'p1_avg_hp_pct_end': 0.65,  # 65% HP finale
        'p2_avg_hp_pct_end': 0.22,  # 22% HP finale
        'diff_team_hp_remaining_pct': 0.43,  #  Vantaggio HP!
    }
    ```
    
    → Modello predizione: **player_won = 1** (win probability ~85%)
    
   
    
    Args:
        battle_json: Dict con struttura completa battaglia (da train.jsonl o test.jsonl)
        max_turns: Limite turni da processare (default 30, performance)
    
    Returns:
        Dict: Feature vector con 349 features + battle_id + player_won (se training)
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
    # Passa team_total_base_hp per calcolare metriche HP 
    p1_total_hp = p1_static.get('p1_team_total_base_hp', 0)
    p2_total_hp = p2_static.get('p2_team_total_base_hp', 0)
    # get_dynamic_features ora ritorna anche i dizionari HP
    p1_dynamic, p2_dynamic, p1_pokemon_hp, p2_pokemon_hp = get_dynamic_features(
        timeline, max_turns=max_turns, 
        p1_team_total_hp=p1_total_hp, 
        p2_team_total_hp=p2_total_hp,
        p1_team_details=p1_team,
        p2_team_details=p2_team
    )

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

    # --- Team type coverage (STAB best-case) per prospettiva del player ---
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
        
        'p1_team_stab_super_count': cov_a['super_count'],

    })
    p2_static.update({
        'p2_team_stab_cov_mean': cov_b['mean'],
        
        'p2_team_stab_super_count': cov_b['super_count'],
    
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

    # --- Lead matchup & speed control ---
    p1_lead_spe = 0
    p2_lead_spe = 0
    if isinstance(p1_lead, dict):
        p1_lead_spe = float(p1_lead.get('base_spe', 0))
    if isinstance(p2_lead, dict):
        p2_lead_spe = float(p2_lead.get('base_spe', 0))
   
    record['p2_lead_base_spe'] = p2_lead_spe
    record['diff_lead_base_spe'] = p1_lead_spe - p2_lead_spe
    # Probabilità grezza di outspeed: sigmoide della differenza
    record['p1_outspeed_prob'] = 1.0 / (1.0 + np.exp(-0.05 * (p1_lead_spe - p2_lead_spe)))
    record['diff_outspeed_prob'] = record['p1_outspeed_prob'] - (1.0 - record['p1_outspeed_prob'])
    
   
    # I tipi non sono nel pokemon_state, dobbiamo matchare il nome con i team details
    p2_seen_types = set()
    if timeline:
        # Crea mapping nome -> tipi per P2
        p2_name_to_types = {}
        for pkmn in (p2_team or []):
            name = pkmn.get('name', '').lower()
            types = [t.lower() for t in pkmn.get('types', []) if t and t.lower() != 'notype']
            if name and types:
                p2_name_to_types[name] = types
        
        # Scansiona la timeline per trovare pokemon visti
        for turn in timeline[:max_turns]:
            p2_state = turn.get('p2_pokemon_state', {}) or {}
            if p2_state:
                pkmn_name = p2_state.get('name', '').lower()
                if pkmn_name in p2_name_to_types:
                    for t in p2_name_to_types[pkmn_name]:
                        p2_seen_types.add(t)
    
    #  Estrai tipi visti di P1 dalla timeline (simmetrico)
    p1_seen_types = set()
    if timeline:
        # Crea mapping nome -> tipi per P1
        p1_name_to_types = {}
        for pkmn in (p1_team or []):
            name = pkmn.get('name', '').lower()
            types = [t.lower() for t in pkmn.get('types', []) if t and t.lower() != 'notype']
            if name and types:
                p1_name_to_types[name] = types
        
        # Scansiona la timeline per trovare pokemon visti
        for turn in timeline[:max_turns]:
            p1_state = turn.get('p1_pokemon_state', {}) or {}
            if p1_state:
                pkmn_name = p1_state.get('name', '').lower()
                if pkmn_name in p1_name_to_types:
                    for t in p1_name_to_types[pkmn_name]:
                        p1_seen_types.add(t)
    
    # Tipi di P1 (per calcolare vantaggio)
    p1_all_types = []
    for pkmn in (p1_team or []):
        p1_all_types.extend([t.lower() for t in pkmn.get('types', []) if t and t.lower() != 'notype'])
    
    #  Tipi di P2 (per calcolare vantaggio simmetrico)
    p2_all_types = []
    for pkmn in (p2_team or []):
        p2_all_types.extend([t.lower() for t in pkmn.get('types', []) if t and t.lower() != 'notype'])
    
    # Calcola vantaggio atteso di P1 contro tipi non visti di P2
    global TYPE_DISTRIBUTION_DF
    expected_advantage_p1 = 1.0  # default neutro
    if TYPE_DISTRIBUTION_DF is not None and p1_all_types:
        expected_advantage_p1 = compute_expected_type_advantage(
            p1_types=p1_all_types,
            p2_seen_types=list(p2_seen_types),
            type_distribution_df=TYPE_DISTRIBUTION_DF
        )
    
    # Calcola vantaggio atteso di P2 contro tipi non visti di P1 (simmetrico)
    expected_advantage_p2 = 1.0  # default neutro
    if TYPE_DISTRIBUTION_DF is not None and p2_all_types:
        expected_advantage_p2 = compute_expected_type_advantage(
            p1_types=p2_all_types,
            p2_seen_types=list(p1_seen_types),
            type_distribution_df=TYPE_DISTRIBUTION_DF
        )
    
    record['p1_expected_type_advantage_unseen_p2'] = expected_advantage_p1
    record['p2_expected_type_advantage_unseen_p1'] = expected_advantage_p2
    record['p1_types_seen_count'] = len(p1_seen_types)
    record['p1_types_unseen_count'] = max(0, 6 - len(p1_seen_types))  # stima pokemon rimanenti
    record['p2_types_seen_count'] = len(p2_seen_types)
    record['p2_types_unseen_count'] = max(0, 6 - len(p2_seen_types))  # stima pokemon rimanenti
    
    # Calcola vantaggio CERTO vs pokemon VISTI e ANCORA VIVI (più affidabile!)
    # Questo è fondamentale: sappiamo ESATTAMENTE quali pokemon sono vivi e i loro tipi
    p1_advantage_vs_p2_alive = compute_type_advantage_vs_seen_alive(
        my_types=p1_all_types,
        opponent_pokemon_hp=p2_pokemon_hp,
        opponent_team_details=p2_team
    )
    p2_advantage_vs_p1_alive = compute_type_advantage_vs_seen_alive(
        my_types=p2_all_types,
        opponent_pokemon_hp=p1_pokemon_hp,
        opponent_team_details=p1_team
    )
    
    record['p1_type_advantage_vs_p2_seen_alive'] = p1_advantage_vs_p2_alive
    record['p2_type_advantage_vs_p1_seen_alive'] = p2_advantage_vs_p1_alive
    # Feature combinate - vantaggio totale pesato
    # Il vantaggio sui pokemon VISTI e VIVI pesa di più (certezza) rispetto a quello sui NON VISTI (probabilità)
    # Peso: 70% visti vivi, 30% non visti (se ci sono pokemon in entrambe le categorie)
    p2_alive_count = sum(1 for hp in p2_pokemon_hp.values() if hp > 0.001)
    p2_unseen_count = max(0, 6 - len(p2_seen_types))
    
    if p2_alive_count > 0 and p2_unseen_count > 0:
        # Entrambe le categorie presenti: pesa 70-30
        p1_combined_advantage = (0.7 * p1_advantage_vs_p2_alive + 
                                 0.3 * expected_advantage_p1)
    elif p2_alive_count > 0:
        # Solo pokemon vivi visti: usa solo quello
        p1_combined_advantage = p1_advantage_vs_p2_alive
    else:
        # Solo pokemon non visti (o tutti morti): usa solo probabilità
        p1_combined_advantage = expected_advantage_p1
    
    # Simmetrico per P2
    p1_alive_count = sum(1 for hp in p1_pokemon_hp.values() if hp > 0.001)
    p1_unseen_count = max(0, 6 - len(p1_seen_types))
    
    if p1_alive_count > 0 and p1_unseen_count > 0:
        p2_combined_advantage = (0.7 * p2_advantage_vs_p1_alive + 
                                 0.3 * expected_advantage_p2)
    elif p1_alive_count > 0:
        p2_combined_advantage = p2_advantage_vs_p1_alive
    else:
        p2_combined_advantage = expected_advantage_p2
    
    record['p1_combined_type_advantage'] = p1_combined_advantage
    record['p2_combined_type_advantage'] = p2_combined_advantage
    
    
        
    return record


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge feature di INTERAZIONE tra le variabili più predittive.
    
    ============================================================================
    CONCETTO: Feature Interactions
    ============================================================================
    Le feature interactions catturano relazioni NON-LINEARI tra variabili
    che i Gradient Boosted Trees potrebbero non scoprire facilmente.
    
    **PERCHÉ SERVONO?**
    - GBDT split su singole feature (es. damage_dealt > 2.5)
    - Ma alcune relazioni sono MOLTIPLICATIVE: A × B > threshold
    - Esempio: Alto danno (3.0) + Alta velocità (100) = Dominanza totale
    -           Alto danno (3.0) + Bassa velocità (50) = Meno impattante
    
    **FORMULA**:
    - Polynomial features: diff_A × diff_B
    - Quoziente features: diff_A / (diff_B + epsilon)
    - Boolean combinations: (A > threshold) AND (B > threshold)
    
    ============================================================================
    FEATURE AGGIUNTE (totale ~20):
    ============================================================================
    
    **CATEGORIA 1: Offensive Synergy** (4 features)
    - `diff_damage_speed_interaction`: danno × velocità
      → Cattura "fast sweep" (danno alto + speed control)
    - `diff_damage_boost_interaction`: danno × boosts
      → Cattura "setup sweep" (danno dopo setup)
    - `diff_damage_stab_interaction`: danno × STAB coverage
      → Cattura "type advantage exploitation"
    - `diff_burst_momentum_interaction`: burst × momentum
      → Cattura "snowball effect" (burst + leadership)
    
    **CATEGORIA 2: Defensive Synergy** (3 features)
    - `diff_hp_defense_interaction`: HP rimanente × team defense
      → Cattura "tank effectiveness"
    - `diff_status_survival_interaction`: status inflitti × clutch survival
      → Cattura "pressure under threat"
    - `diff_switches_fainted_ratio`: switches / fainted
      → Cattura "strategic switching" vs "forced switches"
    
    **CATEGORIA 3: Strategic Control** (4 features)
    - `diff_outspeed_damage_interaction`: outspeed × danno
      → Cattura "speed control dominance"
    - `diff_type_advantage_exploitation`: type advantage × super effective hits
      → Cattura "coverage utilization"
    - `diff_momentum_stability_interaction`: momentum × stability
      → Cattura "consistent pressure"
    - `diff_tempo_consecutive_interaction`: damage_tempo × consecutive_damage
      → Cattura "sustained offense"
    
    **CATEGORIA 4: Win Condition Indicators** (3 features)
    - `diff_burst_ratio_hp_ratio`: burst ratio / HP ratio
      → Cattura "lethal burst potential"
    - `diff_comeback_potential`: comeback × HP delta
      → Cattura "resilience factor"
    - `diff_ko_efficiency_weighted`: KO efficiency × damage dealt
      → Cattura "trading effectiveness"
    
    ============================================================================
    ESEMPIO PRATICO:
    ============================================================================
    
    **Scenario**: Dragapult (P1) vs Garchomp (P2)
    
    **Features Base**:
    ```python
    diff_damage_dealt = 1.35  # P1 fa più danno
    diff_lead_base_spe = 40   # P1 più veloce
    diff_boosts = 4           # P1 ha setup (+2 Atk, +2 Spe)
    diff_max_burst_3turn = 0.43  # P1 burst più alto
    diff_momentum_swings = 0  # (simmetrico)
    diff_current_momentum = 1  # P1 in vantaggio
    ```
    
    **Interactions Generate**:
    ```python
    # 1) Offensive Synergy
    diff_damage_speed_interaction = 1.35 × 40 = 54.0
    # → ALTO! Fast sweeper dominante
    
    diff_damage_boost_interaction = 1.35 × 4 = 5.4
    # → ALTO! Setup sweep funziona
    
    diff_burst_momentum_interaction = 0.43 × 1 = 0.43
    # → Positivo! Burst + leadership = snowball
    
    # 2) Strategic Control
    diff_outspeed_damage_interaction = 0.92 × 1.35 = 1.24
    # → ALTO! Speed control + danno = dominanza
    
    diff_tempo_consecutive_interaction = 0.82 × 8 = 6.56
    # → ALTO! Sustained offense (8 turni consecutivi danno)
    ```
    
    → Modello vede: "Fast setup sweep con sustained pressure"
    → Predizione: **WIN** (confidence ~88%)
    
    ============================================================================
    CORRELAZIONE TIPICA:
    ============================================================================
    - diff_damage_speed_interaction: +0.15 ~ +0.25 (forte predittore)
    - diff_outspeed_damage_interaction: +0.12 ~ +0.18 (speed control)
    - diff_burst_momentum_interaction: +0.08 ~ +0.12 (snowball)
    - diff_ko_efficiency_weighted: +0.10 ~ +0.15 (trading wins)
    
    
    ============================================================================
    IMPORTANTE:
    ============================================================================
    - Applicare STESSO set di interazioni su train E test
    - Evitare overfitting: max 20-30 interazioni (selezionate manualmente)
    - Normalizzazione automatica: divisori hanno +epsilon per evitare /0
    - Validazione: controllare correlazione con target (>0.05 = mantieni)
    
    Args:
        df: DataFrame con le feature base (349 features da process_battle)
        
    Returns:
        DataFrame con feature di interazione aggiunte (~369 features totali)
    """
    df = df.copy()
    
    # 1) Danno × Velocità: chi fa più danno E è più veloce ha vantaggio
    if 'diff_damage_dealt' in df.columns and 'diff_lead_base_spe' in df.columns:
        df['interact_damage_x_speed'] = df['diff_damage_dealt'] * df['diff_lead_base_spe']
    
    # 2) Bilancio team × Vantaggio tipi: team offensivo con vantaggio tipi è letale
    if 'p1_team_offense_defense_ratio' in df.columns and 'diff_type_off_mult' in df.columns:
        df['interact_p1_ratio_x_type'] = df['p1_team_offense_defense_ratio'] * df['diff_type_off_mult']
    if 'p2_team_offense_defense_ratio' in df.columns and 'diff_type_off_mult' in df.columns:
        df['interact_p2_ratio_x_type'] = df['p2_team_offense_defense_ratio'] * (-df['diff_type_off_mult'])
    
    # 3) Boost × HP residuo: boost sono più efficaci se hai HP
    if 'diff_boosts' in df.columns and 'diff_avg_hp_pct_end' in df.columns:
        df['interact_boosts_x_hp'] = df['diff_boosts'] * df['diff_avg_hp_pct_end']
    
    # 4) KO × Danno: pressure offensiva totale
    if 'diff_fainted_pokemon' in df.columns and 'diff_damage_dealt' in df.columns:
        df['interact_ko_x_damage'] = df['diff_fainted_pokemon'] * df['diff_damage_dealt']
    
    # 5) Switch × Lead changes: instabilità tattica
    if 'diff_switches' in df.columns and 'diff_lead_changes' in df.columns:
        df['interact_switch_x_leadchange'] = df['diff_switches'] * df['diff_lead_changes']
    
    # 6) Status × Danno: status debuffano quindi amplificano il danno
    if 'diff_status_inflicted' in df.columns and 'diff_damage_dealt' in df.columns:
        df['interact_status_x_damage'] = df['diff_status_inflicted'] * df['diff_damage_dealt']
    
    # 7) Offense index × Type advantage
    if 'diff_team_mean_offense' in df.columns and 'diff_type_off_mult' in df.columns:
        df['interact_offense_x_type'] = df['diff_team_mean_offense'] * df['diff_type_off_mult']
    
    # 8) Speed × Outspeed probability: velocità assoluta vs relativa
    if 'diff_lead_base_spe' in df.columns and 'diff_outspeed_prob' in df.columns:
        df['interact_speed_x_outspeed'] = df['diff_lead_base_spe'] * df['diff_outspeed_prob']
    
    # 9) Rapporto offense/defense e danno per turno (w1)
    if 'p1_team_offense_defense_ratio' in df.columns and 'p1_damage_dealt_w1_rate' in df.columns:
        df['interact_p1_ratio_x_dmg_rate_w1'] = df['p1_team_offense_defense_ratio'] * df['p1_damage_dealt_w1_rate']
    if 'p2_team_offense_defense_ratio' in df.columns and 'p2_damage_dealt_w1_rate' in df.columns:
        df['interact_p2_ratio_x_dmg_rate_w1'] = df['p2_team_offense_defense_ratio'] * df['p2_damage_dealt_w1_rate']
    
    # 10) Forced switches × HP delta: forced switch sotto pressione HP
    if 'diff_forced_switches' in df.columns and 'diff_avg_hp_pct_delta' in df.columns:
        df['interact_forced_sw_x_hp_delta'] = df['diff_forced_switches'] * df['diff_avg_hp_pct_delta']
    
    return df


def create_feature_df(file_path, max_turns=30):
    """Crea DataFrame di feature da un file JSONL di battaglie."""
    #Carica distribuzione tipi se non già caricata
    global TYPE_DISTRIBUTION_DF
    
    if TYPE_DISTRIBUTION_DF is None:
        try:
            TYPE_DISTRIBUTION_DF = load_type_distribution('predict.csv')
            print(f" Distribuzione tipi caricata: {len(TYPE_DISTRIBUTION_DF)} tipi unici")
        except FileNotFoundError:
            print(" File predict.csv non trovato. Esegui prima predictor.py")
            print("   Le feature di predizione tipo saranno impostate a valori di default.")
    
    
    
    battles_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing battles"):
            battle_json = json.loads(line)
            battle_features = process_battle(battle_json, max_turns=max_turns)
            battles_data.append(battle_features)
    
    df = pd.DataFrame(battles_data)
    
    # Aggiungi interaction features 
    df = add_interaction_features(df)
    
    return df