# 🧠 Pokemon Battle Predictor – Feature Documentation (Versione Finale)

_Aggiornato: Ottobre 2025_  
_Autore: Alessandro Gautieri_  
_Accuratezza complessiva: 85.01%_  
_Feature totali: 349_

---

## ⚙️ Confronto tra Versione Precedente (V2) e Versione Attuale (Finale)

| Aspetto | Versione Precedente (V2) | Versione Attuale (Finale) |
|----------|---------------------------|----------------------------|
| **Struttura delle feature** | Feature base statiche (danno, HP, mosse) | Sistema completo con derivate temporali, differenziali, momenti critici e gestione del tempo |
| **Numero di feature** | 68 | 349 (+281 nuove) |
| **Approccio** | Statistico tradizionale, aggregato per match | Dinamico e temporale, basato su traiettorie, variazioni e ritmo |
| **Ambito** | Singolo evento (colpo o turno) | Contesto globale (ritmo, momentum, burst, vantaggio cumulativo) |
| **Modello ML** | RandomForest | Ensemble CatBoost + LightGBM |
| **Obiettivo** | Stima della vittoria | Analisi strategica del flusso di battaglia |
| **Risultato** | 78.3% accuracy | 85.01% accuracy |

---

## 🧩 Elenco Completo delle Feature (per categoria logica)

### ⚔️ Danno e Combattimento

- `damage_dealt`: quantità totale di danno inflitto durante la battaglia.  
- `physical_damage`: danno inflitto da mosse fisiche.  
- `special_damage`: danno inflitto da mosse speciali.  
- `stab_damage`: danno proveniente da mosse con STAB (Same-Type Attack Bonus).  
- `recoil_damage`: danno autoinflitto da mosse con recoil.  
- `damage_taken`: quantità di danno subita.  
- `damage_ratio`: rapporto tra danno inflitto e danno subito.  
- `avg_damage_per_turn`: danno medio inflitto per turno.  
- `damage_tempo`: danno pesato in base al timing nel turno (ritmo offensivo).  
- `diff_damage_tempo`: differenza di ritmo offensivo tra i due giocatori.  
- `burst_damage_ratio`: proporzione di danno concentrato in brevi finestre.  
- `sustained_pressure_turns`: numero di turni consecutivi con danno elevato.  
- `burst_timing_early`: presenza di picchi di danno nelle fasi iniziali.  
- `burst_timing_late`: presenza di picchi di danno nelle fasi finali.  
- `critical_hits`: numero di colpi critici inflitti.  
- `missed_moves`: numero di mosse fallite.  
- `ko_efficiency`: rapporto tra KO ottenuti e danno totale inflitto.  
- `fainted_pokemon`: numero di Pokémon avversari sconfitti.  
- `diff_fainted_pokemon`: differenza nel numero di KO tra i due giocatori.  
- `hp_lead_duration`: numero di turni con vantaggio in HP rispetto all’avversario.  

---

### ❤️ Gestione HP e Sopravvivenza

- `team_hp_remaining_pct`: percentuale media di HP rimanenti sul team.  
- `avg_hp_per_alive`: HP medio per Pokémon ancora vivo.  
- `hp_when_opponent_critical`: HP residui quando l’avversario è in stato critico (<20%).  
- `avg_hp_pct_delta`: variazione percentuale media di HP nel tempo.  
- `hp_recovery_total`: quantità totale di HP recuperata.  
- `hp_recovery_turns`: numero di turni in cui si è recuperato HP.  
- `hp_lost_turns`: numero di turni con perdita netta di HP.  
- `hp_damage_ratio`: proporzione di HP persi rispetto al totale iniziale.  
- `hp_survival_index`: indice di resistenza media dei Pokémon.  
- `diff_team_hp_remaining_pct`: differenza nel totale HP residuo tra i due team.  
- `diff_avg_hp_per_alive`: differenza di HP medi fra i Pokémon vivi.  

---

### 🌀 Mosse e Precisione

- `moves_used`: numero totale di mosse usate.  
- `unique_moves_used`: numero di mosse distinte utilizzate.  
- `status_moves_used`: numero di mosse non dannose.  
- `attack_moves_used`: numero di mosse offensive.  
- `avg_effectiveness_used`: efficacia media delle mosse usate.  
- `sum_effectiveness_used`: somma dei moltiplicatori di efficacia.  
- `avg_accuracy`: precisione media delle mosse usate.  
- `sum_accuracy`: somma delle precisioni delle mosse usate.  
- `move_success_rate`: tasso di successo delle mosse.  
- `damage_per_move`: danno medio per mossa riuscita.  
- `status_move_rate`: rapporto tra mosse di stato e totali.  
- `high_impact_moves_used`: numero di mosse con potenza > 100.  
- `move_repeat_rate`: frequenza d’uso della stessa mossa.  

---

### 💀 Status e Alterazioni

- `status_inflicted`: numero totale di status inflitti all’avversario.  
- `status_inflicted_brn`: numero di bruciature inflitte.  
- `status_inflicted_psn`: numero di avvelenamenti inflitti.  
- `status_inflicted_slp`: numero di stati di sonno inflitti.  
- `status_inflicted_par`: numero di paralisi inflitte.  
- `status_inflicted_frz`: numero di congelamenti inflitti.  
- `status_turns_inflicted`: turni totali in cui l’avversario è stato affetto da status.  
- `status_received`: numero di status subiti.  
- `status_cleansed`: numero di volte in cui si è rimosso uno status.  
- `diff_status_inflicted`: differenza nel numero di status inflitti.  
- `diff_status_turns_inflicted`: differenza nella durata cumulativa degli status inflitti.  

---

### ⚙️ Boosts e Statistiche

- `boosts_total`: numero totale di potenziamenti ottenuti.  
- `boosts_positive`: numero di aumenti positivi delle statistiche.  
- `boosts_negative`: riduzioni subite.  
- `atk_boosts`, `def_boosts`, `spa_boosts`, `spd_boosts`, `spe_boosts`: conteggi specifici per ogni statistica.  
- `boost_efficiency`: efficacia media dei boost nel contribuire al danno.  
- `boost_timing`: turno medio in cui vengono eseguiti i boost.  
- `diff_boosts_total`: differenza totale di potenziamenti tra giocatori.  

---

### 🧬 Vantaggio di Tipo

- `avg_type_effectiveness`: efficacia media delle mosse in base al tipo.  
- `type_coverage_ratio`: copertura dei tipi nel moveset.  
- `type_matchup_score`: media del vantaggio/svantaggio di tipo.  
- `type_balance_index`: bilanciamento fra tipi offensivi e difensivi.  
- `diff_type_matchup_score`: differenza di vantaggio di tipo tra giocatori.  

---

### 🔄 Switch e Gestione Team

- `switches`: numero totale di cambi Pokémon.  
- `forced_switches`: cambi forzati (es. U-Turn, Roar).  
- `voluntary_switches`: cambi volontari.  
- `avg_turns_before_switch`: media dei turni prima di un cambio.  
- `switch_success_rate`: efficacia dei cambi (se portano vantaggio).  
- `team_coverage_variance`: variazione di tipo e ruolo nel team.  
- `diff_switches`: differenza nel numero di switch effettuati.  

---

### ⏱️ Fasi Temporali (Time Windows)

- `damage_dealt_w1`, `damage_dealt_w2`, `damage_dealt_w3`: danno medio nelle fasi iniziali, medie e finali.  
- `boosts_w1`, `boosts_w2`, `boosts_w3`: numero di boost per fase.  
- `status_inflicted_w1`, `status_inflicted_w2`, `status_inflicted_w3`: status inflitti per finestra temporale.  
- `momentum_w1`, `momentum_w2`, `momentum_w3`: variazioni di controllo nelle diverse fasi.  
- `hp_trend_w1`, `hp_trend_w2`, `hp_trend_w3`: andamento medio HP nel tempo.  

---

### ⚡ Momentum e Controllo del Flusso

- `momentum_swings`: numero di variazioni di leadership HP.  
- `current_momentum`: valore di momentum corrente.  
- `momentum_stability`: costanza del vantaggio nel tempo.  
- `favorable_momentum_swings`: swing positivi (vantaggi a proprio favore).  
- `momentum_ratio`: rapporto fra swing favorevoli e totali.  
- `diff_momentum_stability`: differenza di stabilità del controllo tra giocatori.  
- `diff_current_momentum`: differenza di momentum istantaneo.  

---

### 🔥 Momenti Critici e Clutch Play

- `clutch_survival_turns`: turni sopravvissuti con HP < 20%.  
- `max_burst_3turn`: massimo danno inflitto in 3 turni consecutivi.  
- `opponent_comeback_count`: volte in cui l’avversario ha dovuto rimontare.  
- `comeback_potential`: probabilità di ribaltamento stimata dal modello.  
- `critical_turns_won`: turni critici (clutch) vinti.  
- `critical_turns_lost`: turni critici persi.  
- `critical_event_density`: densità di eventi critici nel match.  

---

### 🧠 Statistiche Derivate e Differenziali

Tutte le feature sopra esistono in doppia versione (`p1_`, `p2_`) e in versione differenziale `diff_*` calcolata come `p1 - p2`.

Esempi:
- `diff_damage_dealt`
- `diff_status_inflicted`
- `diff_team_hp_remaining_pct`
- `diff_boosts_total`
- `diff_momentum_stability`

---

## 📊 Tabella delle Feature Più Importanti

| Feature | Descrizione | Correlazione/Importanza |
|----------|-------------|--------------------------|
| `diff_damage_tempo` | Differenza nel ritmo offensivo complessivo | **+0.622** |
| `diff_status_inflicted` | Vantaggio sugli status inflitti | **+0.595** |
| `diff_team_hp_remaining_pct` | Vantaggio medio di HP sul team | **+0.542** |
| `diff_damage_dealt` | Vantaggio in danno inflitto totale | **+0.540** |
| `diff_fainted_pokemon` | Differenza nel numero di Pokémon sconfitti | **+0.476** |
| `favorable_momentum_swings` | Swing positivi di controllo partita | **+0.140** |
| `opponent_comeback_count` | Rimonta subita dall’avversario | **+0.427** |
| `clutch_survival_turns` | Turni sopravvissuti in stato critico | **+0.391** |
| `burst_damage_ratio` | Concentrazione di danno in brevi finestre | **+0.368** |

---

## 🧾 Sintesi Finale

La versione attuale del modello integra tutte le dimensioni del combattimento Pokémon:  
- **danno**, **status**, **boost**, **momentum**, **sopravvivenza** e **tempo**;  
- tutte le feature sono calcolate **per giocatore**, **per fase temporale** e **in forma differenziale**;  
- l’approccio combina **statistiche statiche** e **metriche dinamiche derivate**, offrendo una rappresentazione completa del flusso di battaglia.

Risultato: un sistema predittivo robusto, interpretabile e adatto a generalizzare su battaglie di ogni tipo.

---

