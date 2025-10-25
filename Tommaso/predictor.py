# predictor.py
# ============================================================================
# MODULO: Analisi e Predizione Tipi Pokemon
# ============================================================================
# 
# SCOPO:
# Questo modulo implementa un sistema probabilistico per predire i tipi dei
# pokemon NON ancora visti nel team avversario basandosi su:
# 1. Distribuzione statistica dei tipi nel metagame (da training set)
# 2. Tipi già osservati durante la battaglia
# 3. Calcolo bayesiano delle probabilità dei tipi rimanenti
#
# UTILIZZO:
# - Durante feature engineering, calcola "expected type advantage" vs tipi non visti
# - Stima quali tipi è più probabile incontrare nei pokemon non ancora rivelati
# - Fornisce stats medie attese (HP, ATK, ecc.) dei pokemon rimanenti
#
# FUNZIONI PRINCIPALI:
# - build_type_distribution_csv(): Analizza train.jsonl e crea distribuzione tipi
# - predict_unseen_types(): Predice tipi mancanti dato ciò che abbiamo visto
# - load_type_distribution(): Carica CSV con distribuzione pre-calcolata
#
# ============================================================================

import json
import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple
from tqdm import tqdm


def build_type_distribution_csv(train_file_path: str, output_csv: str = 'predict.csv') -> None:
    """
    Analizza il dataset di training per costruire la distribuzione statistica dei tipi Pokemon.
    
    FUNZIONAMENTO:
    1. Legge train.jsonl riga per riga (ogni battaglia)
    2. Esamina SOLO i team di P1 (considerati competitivi/rappresentativi del metagame)
    3. Conta:
       - Frequenza di ogni tipo singolo (es. 'water', 'fire')
       - Frequenza di ogni coppia di tipi per pokemon dual-type (es. ('water', 'flying'))
    4. Calcola probabilità empiriche: P(tipo) = count(tipo) / total_types_seen
    5. Salva risultati in CSV con due sezioni separate
    
    PERCHÉ SOLO P1?
    - P1 è sempre il giocatore con team "reale" (6 pokemon noti)
    - P2 nel training ha solo il lead (1 pokemon), quindi statistiche distorte
    - Assumiamo che P1 rappresenti la distribuzione "vera" del metagame
    
    OUTPUT CSV:
    - Sezione 1: Tipi singoli con [type, count, probability, percentage]
    - Sezione 2: Coppie di tipi con [type1, type2, count, probability, percentage]
    
    Returns:
        Tuple[DataFrame, DataFrame]: (df_tipi_singoli, df_coppie_tipi)
    """
    print(f"🔍 Analizzando distribuzione tipi da {train_file_path}...")
    
    # Contatori
    single_type_counts = Counter()  # Conta tipi singoli
    type_pair_counts = Counter()    # Conta coppie di tipi (ordinati alfabeticamente)
    total_pokemon = 0
    total_types_seen = 0
    
    # Leggi il file JSONL
    with open(train_file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processando battaglie"):
            battle = json.loads(line)
            
            # Esamina solo P1 (giocatore con team competitivo)
            p1_team = battle.get('p1_team_details', [])
            
            for pokemon in p1_team:
                total_pokemon += 1
                types = pokemon.get('types', [])
                
                # Filtra 'notype' e None
                valid_types = [t.lower() for t in types if t and t.lower() != 'notype']
                
                if not valid_types:
                    continue
                
                # Conta ogni tipo singolo
                for poke_type in valid_types:
                    single_type_counts[poke_type] += 1
                    total_types_seen += 1
                
                # Se è dual-type, conta anche la coppia
                if len(valid_types) == 2:
                    # Ordina alfabeticamente per consistenza
                    type_pair = tuple(sorted(valid_types))
                    type_pair_counts[type_pair] += 1
    
    print(f"\n📊 Statistiche:")
    print(f"   - Pokemon totali analizzati: {total_pokemon}")
    print(f"   - Tipi totali visti: {total_types_seen}")
    print(f"   - Tipi unici: {len(single_type_counts)}")
    print(f"   - Coppie di tipi uniche: {len(type_pair_counts)}")
    
    # Crea DataFrame per tipi singoli
    type_data = []
    for poke_type, count in single_type_counts.most_common():
        probability = count / total_types_seen
        type_data.append({
            'type': poke_type,
            'count': count,
            'probability': probability,
            'percentage': probability * 100
        })
    
    df_types = pd.DataFrame(type_data)
    
    # Crea DataFrame per coppie di tipi
    pair_data = []
    for type_pair, count in type_pair_counts.most_common():
        probability = count / total_pokemon  # Probabilità su tutti i pokemon
        pair_data.append({
            'type1': type_pair[0],
            'type2': type_pair[1],
            'count': count,
            'probability': probability,
            'percentage': probability * 100
        })
    
    df_pairs = pd.DataFrame(pair_data)
    
    # Salva in CSV (due sezioni separate)
    with open(output_csv, 'w', encoding='utf-8') as f:
        f.write("# DISTRIBUZIONE TIPI SINGOLI\n")
        df_types.to_csv(f, index=False)
        f.write("\n# DISTRIBUZIONE COPPIE DI TIPI\n")
        df_pairs.to_csv(f, index=False)
    
    print(f"\n✅ Distribuzione salvata in {output_csv}")
    print(f"\n🔝 Top 10 tipi più comuni:")
    for idx, row in df_types.head(10).iterrows():
        print(f"   {idx+1}. {row['type']:12s} - {row['percentage']:5.2f}% ({row['count']} occorrenze)")
    
    return df_types, df_pairs


def build_type_stats_distribution(train_file_path: str) -> Dict[str, Dict[str, float]]:
    """
    Analizza il dataset di training per costruire la distribuzione delle stats per tipo.
    
    Returns:
        Dict[tipo, Dict[stat, valore_medio]]
        Es: {'water': {'hp': 75.3, 'atk': 68.1, ...}, 'fire': {...}, ...}
    """
    print(f"🔍 Analizzando distribuzione stats per tipo da {train_file_path}...")
    
    # Dizionario: tipo -> lista di stats
    type_to_stats = {}
    
    with open(train_file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processando pokemon per stats"):
            battle = json.loads(line)
            p1_team = battle.get('p1_team_details', [])
            
            for pokemon in p1_team:
                types = pokemon.get('types', [])
                valid_types = [t.lower() for t in types if t and t.lower() != 'notype']
                
                if not valid_types:
                    continue
                
                # Estrai stats
                stats = {
                    'hp': pokemon.get('base_hp', 0),
                    'atk': pokemon.get('base_atk', 0),
                    'def': pokemon.get('base_def', 0),
                    'spa': pokemon.get('base_spa', 0),
                    'spd': pokemon.get('base_spd', 0),
                    'spe': pokemon.get('base_spe', 0)
                }
                
                # Aggiungi stats per ogni tipo del pokemon
                for poke_type in valid_types:
                    if poke_type not in type_to_stats:
                        type_to_stats[poke_type] = []
                    type_to_stats[poke_type].append(stats)
    
    # Calcola medie per tipo
    type_stats_avg = {}
    for poke_type, stats_list in type_to_stats.items():
        if not stats_list:
            continue
        
        avg_stats = {
            'hp': np.mean([s['hp'] for s in stats_list]),
            'atk': np.mean([s['atk'] for s in stats_list]),
            'def': np.mean([s['def'] for s in stats_list]),
            'spa': np.mean([s['spa'] for s in stats_list]),
            'spd': np.mean([s['spd'] for s in stats_list]),
            'spe': np.mean([s['spe'] for s in stats_list]),
            'count': len(stats_list)
        }
        avg_stats['bst'] = sum([avg_stats[s] for s in ['hp', 'atk', 'def', 'spa', 'spd', 'spe']])
        avg_stats['offense'] = avg_stats['atk'] + avg_stats['spa']
        avg_stats['defense'] = avg_stats['hp'] + avg_stats['def'] + avg_stats['spd']
        
        type_stats_avg[poke_type] = avg_stats
    
    print(f"\n✅ Stats medie calcolate per {len(type_stats_avg)} tipi")
    print(f"\n📊 Sample - Stats medie per 'water':")
    if 'water' in type_stats_avg:
        for stat, val in type_stats_avg['water'].items():
            print(f"   {stat:8s}: {val:6.1f}")
    
    return type_stats_avg


def estimate_unseen_pokemon_stats(unseen_types_probs: Dict[str, float],
                                   type_stats_dist: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Stima le stats medie dei pokemon non visti basandosi sulla probabilità dei tipi.
    
    Args:
        unseen_types_probs: Dict[tipo, probabilità] dei tipi non visti
        type_stats_dist: Dict[tipo, Dict[stat, valore]] delle stats medie per tipo
    
    Returns:
        Dict[stat, valore stimato] - stats stimate pesate per probabilità
    """
    if not unseen_types_probs:
        return {'bst': 0, 'offense': 0, 'defense': 0, 'hp': 0, 'atk': 0, 'def': 0, 'spa': 0, 'spd': 0, 'spe': 0}
    
    estimated_stats = {'bst': 0.0, 'offense': 0.0, 'defense': 0.0, 'hp': 0.0, 
                      'atk': 0.0, 'def': 0.0, 'spa': 0.0, 'spd': 0.0, 'spe': 0.0}
    
    total_prob = sum(unseen_types_probs.values())
    if total_prob == 0:
        return estimated_stats
    
    # Weighted average basato su probabilità tipi
    for poke_type, prob in unseen_types_probs.items():
        if poke_type in type_stats_dist:
            weight = prob / total_prob
            type_stats = type_stats_dist[poke_type]
            for stat in estimated_stats:
                estimated_stats[stat] += weight * type_stats.get(stat, 0.0)
    
    return estimated_stats


def predict_unseen_types(seen_types: List[str], 
                         type_distribution: pd.DataFrame,
                         team_size: int = 6) -> Dict[str, float]:
    """
    Predice i tipi "mancanti" del team avversario basandosi sui tipi già visti.
    
    ALGORITMO (Bayesiano Semplificato):
    1. Conta quanti pokemon abbiamo già visto (stima: 1 tipo = 1 pokemon)
    2. Calcola quanti pokemon rimangono da vedere: unseen = team_size - seen
    3. Per ogni tipo NON visto, calcola probabilità che appaia almeno 1 volta:
       
       P(tipo appare) = 1 - P(tipo NON appare in NESSUNO dei rimanenti)
       P(tipo NON appare) = (1 - base_prob)^unseen_pokemon
       
       dove base_prob = frequenza del tipo nel metagame (da distribuzione)
    
    ESEMPIO PRATICO:
    - Visti: ['water', 'fire', 'grass'] → 3 pokemon visti
    - Rimangono: 6 - 3 = 3 pokemon
    - Tipo 'dragon' ha base_prob = 0.08 (8% nel metagame)
    - P(dragon appare) = 1 - (1 - 0.08)³ = 1 - 0.778 = 0.222 (22.2%)
    
    INTUIZIONE:
    - Tipi comuni (es. water 18%) → Alta probabilità di apparire
    - Tipi rari (es. ice 3%) → Bassa probabilità
    - Più pokemon rimangono, più è probabile vedere tipi comuni
    
    UTILIZZO IN FEATURE ENGINEERING:
    Questo viene usato per calcolare "expected type advantage":
    - Calcolo vantaggio di tipo del mio team vs ogni tipo possibile
    - Peso per probabilità: advantage = Σ(prob[tipo] × mult[tipo])
    - Risultato: vantaggio atteso contro team NON ancora visto
    
    Args:
        seen_types: Lista dei tipi già visti di P2 (es. ['water', 'fire'])
        type_distribution: DataFrame con colonne ['type', 'probability']
        team_size: Dimensione del team (default 6 per battles standard)
    
    Returns:
        Dict[tipo: probabilità] per i tipi non ancora visti
        Esempio: {'dragon': 0.222, 'electric': 0.315, ...}
    """
    # Normalizza seen_types
    seen_types_set = set(t.lower() for t in seen_types if t and t.lower() != 'notype')
    
    # Stima numero pokemon visti (approssimazione: un tipo per pokemon)
    # In realtà potremmo avere dual-type, quindi è una stima conservativa
    pokemon_seen = len(seen_types_set)
    pokemon_unseen = max(0, team_size - pokemon_seen)
    
    if pokemon_unseen == 0:
        return {}  # Nessun pokemon da predire
    
    # Per ogni tipo non visto, calcola probabilità
    unseen_type_probs = {}
    
    for _, row in type_distribution.iterrows():
        poke_type = row['type']
        base_prob = row['probability']
        
        if poke_type not in seen_types_set:
            # Probabilità che questo tipo appaia almeno una volta nei pokemon rimanenti
            # P(tipo appare) = 1 - P(tipo non appare in nessuno dei rimanenti)
            # P(tipo non appare) = (1 - base_prob)^pokemon_unseen
            prob_appears = 1 - (1 - base_prob) ** pokemon_unseen
            unseen_type_probs[poke_type] = prob_appears
    
    return unseen_type_probs


def load_type_distribution(csv_path: str = 'predict.csv') -> pd.DataFrame:
    """
    Carica la distribuzione dei tipi dal CSV generato.
    
    Args:
        csv_path: Path al CSV con la distribuzione
    
    Returns:
        DataFrame con la distribuzione dei tipi singoli
    """
    # Leggi solo la sezione dei tipi singoli (prima del secondo header)
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Trova dove inizia la sezione dei tipi singoli
    type_section_start = None
    type_section_end = None
    
    for i, line in enumerate(lines):
        if line.startswith('# DISTRIBUZIONE TIPI SINGOLI'):
            type_section_start = i + 1
        elif line.startswith('# DISTRIBUZIONE COPPIE'):
            type_section_end = i
            break
    
    # Leggi solo la sezione dei tipi singoli
    type_lines = lines[type_section_start:type_section_end]
    
    # Crea DataFrame
    from io import StringIO
    df = pd.read_csv(StringIO(''.join(type_lines)))
    
    return df

