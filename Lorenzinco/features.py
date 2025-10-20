# Compact features + priors-driven enrichment WITH PROGRESS + "alive@30" aggregates.
import pandas as pd, numpy as np, json
from typing import Dict, Any, List, Tuple
from collections import Counter
from tqdm import tqdm
import config
from priors import load_priors

PRIORS = load_priors()

TYPE_CHART = {
  'fire':{'grass':2,'ice':2,'bug':2,'steel':2,'water':.5,'rock':.5,'dragon':.5,'fire':.5},
  'water':{'fire':2,'ground':2,'rock':2,'water':.5,'grass':.5,'dragon':.5},
  'grass':{'water':2,'ground':2,'rock':2,'fire':.5,'grass':.5,'poison':.5,'flying':.5,'bug':.5,'dragon':.5,'steel':.5},
  'electric':{'water':2,'flying':2,'grass':.5,'electric':.5,'dragon':.5,'ground':0},
  'ice':{'grass':2,'ground':2,'flying':2,'dragon':2,'fire':.5,'water':.5,'ice':.5,'steel':.5},
  'fighting':{'normal':2,'ice':2,'rock':2,'dark':2,'steel':2,'poison':.5,'flying':.5,'psychic':.5,'bug':.5,'fairy':.5,'ghost':0},
  'ground':{'fire':2,'electric':2,'poison':2,'rock':2,'steel':2,'grass':.5,'bug':.5,'flying':0},
  'flying':{'grass':2,'fighting':2,'bug':2,'electric':.5,'rock':.5,'steel':.5},
  'psychic':{'fighting':2,'poison':2,'psychic':.5,'steel':.5,'dark':0},
  'bug':{'grass':2,'psychic':2,'dark':2,'fire':.5,'fighting':.5,'poison':.5,'flying':.5,'ghost':.5,'steel':.5,'fairy':.5},
  'rock':{'fire':2,'ice':2,'flying':2,'bug':2,'fighting':.5,'ground':.5,'steel':.5},
  'ghost':{'psychic':2,'ghost':2,'dark':.5,'normal':0},
  'dragon':{'dragon':2,'steel':.5,'fairy':0},
  'dark':{'psychic':2,'ghost':2,'fighting':.5,'dark':.5,'fairy':.5},
  'steel':{'ice':2,'rock':2,'fairy':2,'fire':.5,'water':.5,'electric':.5,'steel':.5},
  'fairy':{'fighting':2,'dragon':2,'dark':2,'fire':.5,'poison':.5,'steel':.5},
  'normal':{'rock':.5,'ghost':0,'steel':.5},
  'poison':{'grass':2,'fairy':2,'poison':.5,'ground':.5,'rock':.5,'ghost':.5,'steel':0},
}

def type_multiplier(attacking_types, defending_types):
    mult=1.0
    for atk in (attacking_types or []):
        row = TYPE_CHART.get((atk or '').lower(), {})
        for df in (defending_types or []):
            mult *= float(row.get((df or '').lower(), 1.0))
    return mult

# -------------------- Static team features (trimmed: removed *_max_*) --------------------
def get_static_features(team_details, prefix):
    """Averages/variability + diversity/entropy; NO *_max_* anymore."""
    features={}; stats=['base_hp','base_atk','base_def','base_spa','base_spd','base_spe']
    all_stats={s:[] for s in stats}; all_types=[]; off=[]; deff=[]; spe=[]
    for p in (team_details or []):
        for s in stats: all_stats[s].append(p.get(s,0))
        all_types += (p.get('types') or [])
        off.append(p.get('base_atk',0)+p.get('base_spa',0))
        deff.append(p.get('base_hp',0)+p.get('base_def',0)+p.get('base_spd',0))
        spe.append(p.get('base_spe',0))
    for s,vals in all_stats.items():
        features[f'{prefix}_team_mean_{s}']=float(np.mean(vals)) if vals else 0.0
        features[f'{prefix}_team_std_{s}']=float(np.std(vals)) if vals else 0.0
    def _agg(name, arr):
        features[f'{prefix}_team_mean_{name}']=float(np.mean(arr)) if arr else 0.0
        features[f'{prefix}_team_std_{name}']=float(np.std(arr)) if arr else 0.0
    _agg('offense',off); _agg('defense',deff); _agg('speed',spe)
    denom=features[f'{prefix}_team_mean_defense'] or 1.0
    features[f'{prefix}_team_offense_defense_ratio']=features[f'{prefix}_team_mean_offense']/denom
    # Types
    types=[t for t in all_types if t!='notype']
    features[f'{prefix}_team_unique_types']=len(set(types))
    if types:
        _,cnt=np.unique(types,return_counts=True); p=cnt/cnt.sum()
        features[f'{prefix}_team_type_entropy']=float(-(p*np.log(p+1e-12)).sum())
    else:
        features[f'{prefix}_team_type_entropy']=0.0
    return features

# -------------------- Dynamic features (trimmed: removed time_to_first_ko_inflicted) --------------------
def get_dynamic_features(battle_log, max_turns=30):
    base=['damage_dealt','boosts','fainted_pokemon','switches','status_inflicted','priority_moves']
    def initd():
        d={m:0 for m in base}
        for m in base: d[f'{m}_w1']=d[f'{m}_w2']=d[f'{m}_w3']=0
        d.update({'status_turns_inflicted':0,'ema_damage_dealt':0.0,
                  **{f'boosts_{s}_sum':0 for s in ['atk','def','spa','spd','spe']},
                  'lead_changes':0,'forced_switches':0})
        return d
    p1,p2=initd(),initd(); p1_last_hp={}; p2_last_hp={}; p1_last=None; p2_last=None
    p1_ol_status = {}; p2_ol_status = {}
    if not battle_log: return p1,p2
    n=min(max_turns,len(battle_log)); p1_cum=p2_cum=0.0; last=0; prev_dmg_p1=prev_dmg_p2=0.0
    def _hp(x):
        try: v=float(x)
        except: v=0.0
        return max(0.0,min(100.0,v))
    for i,t in enumerate(battle_log[:max_turns]):
        w='w1' if i<10 else ('w2' if i<20 else 'w3')
        s1,s2=(t.get('p1_pokemon_state') or {}),(t.get('p2_pokemon_state') or {})
        m1,m2=(t.get('p1_move_details') or {}),(t.get('p2_move_details') or {})
        if not (s1 and s2): continue
        n1,n2=s1.get('name'),s2.get('name')
        if p1_last and n1!=p1_last: p1['switches']+=1; p1[f'switches_{w}']+=1;  p1['forced_switches']+= int(prev_dmg_p1>=15)
        if p2_last and n2!=p2_last: p2['switches']+=1; p2[f'switches_{w}']+=1;  p2['forced_switches']+= int(prev_dmg_p2>=15)
        p1_last,p2_last=n1,n2
        st2=(s2.get('status','nostatus') or 'nostatus'); st1=(s1.get('status','nostatus') or 'nostatus')
        if n2 and st2!='nostatus' and (p1_ol_status.get(n2,'nostatus')=='nostatus'): p1['status_inflicted']+=1; p1[f'status_inflicted_{w}']+=1
        if n1 and st1!='nostatus' and (p2_ol_status.get(n1,'nostatus')=='nostatus'): p2['status_inflicted']+=1; p2[f'status_inflicted_{w}']+=1
        if n2: p1_ol_status[n2]=st2
        if n1: p2_ol_status[n1]=st1
        if st2!='nostatus': p1['status_turns_inflicted']+=1
        if st1!='nostatus': p2['status_turns_inflicted']+=1
        if m1.get('priority',0)>0: p1['priority_moves']+=1; p1[f'priority_moves_{w}']+=1
        if m2.get('priority',0)>0: p2['priority_moves']+=1; p2[f'priority_moves_{w}']+=1
        hp2=_hp(s2.get('hp_pct',0)); d1=0
        if n2 in p1_last_hp:
            d1=p1_last_hp[n2]-hp2
            if d1>0: p1['damage_dealt']+=d1; p1[f'damage_dealt_{w}']+=d1
        p1_last_hp[n2]=hp2
        p1['ema_damage_dealt']=0.3*max(d1,0)+0.7*p1['ema_damage_dealt']
        hp1=_hp(s1.get('hp_pct',0)); d2=0
        if n1 in p2_last_hp:
            d2=p2_last_hp[n1]-hp1
            if d2>0: p2['damage_dealt']+=d2; p2[f'damage_dealt_{w}']+=d2
        p2_last_hp[n1]=hp1
        p2['ema_damage_dealt']=0.3*max(d2,0)+0.7*p2['ema_damage_dealt']
        b1,b2=s1.get('boosts',{}) or {}, s2.get('boosts',{}) or {}
        s1sum=sum(v for v in b1.values() if isinstance(v,(int,float)) and v>0)
        s2sum=sum(v for v in b2.values() if isinstance(v,(int,float)) and v>0)
        p1['boosts']+=s1sum; p1[f'boosts_{w}']+=s1sum
        p2['boosts']+=s2sum; p2[f'boosts_{w}']+=s2sum
        for st in ['atk','def','spa','spd','spe']:
            v=b1.get(st,0);  p1[f'boosts_{st}_sum']+= v if isinstance(v,(int,float)) and v>0 else 0
            v=b2.get(st,0);  p2[f'boosts_{st}_sum']+= v if isinstance(v,(int,float)) and v>0 else 0
        p1_cum+=max(d1,0); p2_cum+=max(d2,0)
        leader=1 if p1_cum>p2_cum else (-1 if p2_cum>p1_cum else 0)
        if (leader and leader!=last) and last:
            p1['lead_changes']+=1; p2['lead_changes']+=1
        if leader: last=leader
        prev_dmg_p2=max(d1,0); prev_dmg_p1=max(d2,0)
    for m in base:
        p1[f'{m}_per_turn']=p1[m]/n if n else 0
        p2[f'{m}_per_turn']=p2[m]/n if n else 0
    return p1,p2

# -------------------- Priors-driven enrichment --------------------
def _entropy(prob_list: List[float]) -> float:
    p = np.asarray(prob_list, dtype=float)
    p = p[p>0]
    return float(-(p*np.log(p)).sum()) if p.size else 0.0

def _infer_p2_from_priors(bj: Dict[str,Any]) -> Dict[str,float]:
    cooc = PRIORS.get("cooc_prob", {})
    bmean = PRIORS.get("base_stats_mean", {})
    lead = (bj.get("p2_lead_details") or {}).get("name","")
    lead = (lead or "").lower()
    seen = []
    for t in bj.get("battle_timeline", bj.get("battle_log", [])) or []:
        nm = (t.get("p2_pokemon_state") or {}).get("name") or ""
        nm = nm.lower()
        if nm: seen.append(nm)
    seeds = [lead] + seen
    cand = Counter()
    for s in seeds:
        for k,pr in (cooc.get(s, {}) or {}).items():
            cand[k] += pr
    for s in seeds: cand.pop(s, None)
    if not cand:
        return {"p2_infer_mass_top1":0.0,"p2_infer_mass_top3":0.0,"p2_infer_entropy":0.0,"p2_infer_expected_base_spe":0.0}
    tot = sum(cand.values())
    probs = {k:v/tot for k,v in cand.items()} if tot>0 else {}
    sorted_probs = sorted(probs.values(), reverse=True)
    mass_top1 = sorted_probs[0] if sorted_probs else 0.0
    mass_top3 = sum(sorted_probs[:3]) if sorted_probs else 0.0
    ent = _entropy(sorted_probs)
    exp_spe = 0.0
    for mon, pr in probs.items():
        exp_spe += pr * float((bmean.get(mon, {}) or {}).get("base_spe", 0.0))
    return {
        "p2_infer_mass_top1": float(mass_top1),
        "p2_infer_mass_top3": float(mass_top3),
        "p2_infer_entropy": float(ent),
        "p2_infer_expected_base_spe": float(exp_spe),
    }

def _lead_move_type_priors(side_lead: dict) -> Dict[str,float]:
    mprobs = PRIORS.get("move_type_prob", {})
    name = (side_lead or {}).get("name","")
    name = (name or "").lower()
    d = mprobs.get(name, {})
    if not d:
        return {"lead_move_type_entropy":0.0,"lead_move_same_type_mass":0.0}
    lead_types = set([t.lower() for t in (side_lead.get("types") or []) if t and t!="notype"])
    same_type_mass = sum([p for t,p in d.items() if t in lead_types])
    return {
        "lead_move_type_entropy": _entropy(list(d.values())),
        "lead_move_same_type_mass": float(same_type_mass),
    }

# -------------------- NEW: Alive-team aggregates at turn 30 --------------------
def _collect_alive(side: str, bj: Dict[str,Any], max_turns: int=30) -> Dict[str, Tuple[float,str]]:
    """
    Return {mon_name_lower: (last_hp_pct, last_status)} for 'p1' or 'p2' after first max_turns.
    """
    last_hp: Dict[str,float] = {}
    last_status: Dict[str,str] = {}
    log = bj.get("battle_timeline", bj.get("battle_log", [])) or []
    for t in log[:max_turns]:
        st = (t.get(f"{side}_pokemon_state") or {})
        name = (st.get("name") or "").lower()
        if not name: continue
        # hp_pct may be 0..1 or 0..100 depending on producer; clamp later
        hp = st.get("hp_pct", None)
        if hp is not None:
            try: v = float(hp)
            except: v = 0.0
            # normalize to 0..100
            v = v*100.0 if v<=1.0 else v
            v = max(0.0, min(100.0, v))
            last_hp[name] = v
        last_status[name] = (st.get("status") or "nostatus")
    # Alive if hp>0 and status not explicit faint
    return {k:(hp,last_status.get(k,"nostatus")) for k,hp in last_hp.items() if hp>0 and (last_status.get(k,"nostatus")!="fnt")}

def _alive_aggregates(side: str, bj: Dict[str,Any], max_turns: int=30) -> Dict[str,float]:
    alive = _collect_alive(side, bj, max_turns=max_turns)
    prefix = f"{side}_alive30"
    bmean = PRIORS.get("base_stats_mean", {})
    # Build per-mon stat vectors from priors; if missing, try from team_details
    # fallback: zeros
    # Map team details for quick lookup
    team_map = {}
    team_key = "p1_team_details" if side=="p1" else "p2_team_details"
    for p in (bj.get(team_key) or []):
        nm = (p.get("name") or "").lower()
        if nm: team_map[nm] = p

    stats = ["base_hp","base_atk","base_def","base_spa","base_spd","base_spe"]
    rows = []
    weights = []
    for mon,(hp,status) in alive.items():
        bs = bmean.get(mon)
        if not bs:
            bs = team_map.get(mon, {})
        vec = [float(bs.get(s,0.0)) for s in stats]
        rows.append(vec)
        weights.append(hp/100.0)  # hp-weight in [0,1]
    feats = {}
    cnt = len(rows)
    feats[f"{prefix}_count"] = float(cnt)
    if cnt==0:
        # zeros for all sums/means
        for s in stats:
            feats[f"{prefix}_sum_{s}"] = 0.0
            feats[f"{prefix}_mean_{s}"] = 0.0
            feats[f"{prefix}_hpsum_{s}"] = 0.0
        feats[f"{prefix}_sum_offense"] = feats[f"{prefix}_hpsum_offense"] = 0.0
        feats[f"{prefix}_sum_defense"] = feats[f"{prefix}_hpsum_defense"] = 0.0
        feats[f"{prefix}_sum_speed"]   = feats[f"{prefix}_hpsum_speed"]   = 0.0
        return feats

    M = np.asarray(rows, dtype=float)
    w = np.asarray(weights, dtype=float)
    # Plain sums/means
    sums = M.sum(axis=0)
    means = M.mean(axis=0)
    for j,s in enumerate(stats):
        feats[f"{prefix}_sum_{s}"] = float(sums[j])
        feats[f"{prefix}_mean_{s}"] = float(means[j])
    # HP-weighted sums
    wsums = (M * w[:,None]).sum(axis=0)
    for j,s in enumerate(stats):
        feats[f"{prefix}_hpsum_{s}"] = float(wsums[j])
    # Composite indices
    atk, spa = M[:,1], M[:,3]
    hp, de, sd = M[:,0], M[:,2], M[:,4]
    spe = M[:,5]
    feats[f"{prefix}_sum_offense"] = float((atk+spa).sum())
    feats[f"{prefix}_sum_defense"] = float((hp+de+sd).sum())
    feats[f"{prefix}_sum_speed"]   = float(spe.sum())
    feats[f"{prefix}_hpsum_offense"] = float(((atk+spa)*w).sum())
    feats[f"{prefix}_hpsum_defense"] = float(((hp+de+sd)*w).sum())
    feats[f"{prefix}_hpsum_speed"]   = float((spe*w).sum())
    return feats

# -------------------- Battle processing --------------------
def process_battle(bj, max_turns=30):
    rec={'battle_id':bj.get('battle_id')}
    if 'player_won' in bj: rec['player_won']=bj['player_won']

    # Static (trimmed)
    p1s=get_static_features(bj.get('p1_team_details',[]),'p1')
    p2t=bj.get('p2_team_details')
    if not p2t:
        lead=bj.get('p2_lead_details') or {}
        p2t=[lead] if lead else []
    p2s=get_static_features(p2t,'p2')

    # Dynamics (trimmed)
    tl=bj.get('battle_log', bj.get('battle_timeline',[]))
    p1d,p2d=get_dynamic_features(tl, max_turns=max_turns)

    # Perspective normalization
    player_is_p1 = bool(bj.get('player_is_p1')) if 'player_is_p1' in bj else (str(bj.get('player_side','p1')).lower()=='p1')
    if not player_is_p1: p1s,p2s=p2s,p1s; p1d,p2d=p2d,p1d

    # Lead/type & speed features
    def _ttypes(team):
        t=[]
        for p in team or []: t += (p.get('types') or [])
        return list(set([x for x in t if x and x!='notype']))
    p1_lead=bj.get('p1_lead_details') or (bj.get('battle_timeline',[{}])[0].get('p1_pokemon_state') if bj.get('battle_timeline') else None)
    p2_lead=bj.get('p2_lead_details') or (bj.get('battle_timeline',[{}])[0].get('p2_pokemon_state') if bj.get('battle_timeline') else None)
    p1t=(p1_lead.get('types') if isinstance(p1_lead,dict) else []) or _ttypes(bj.get('p1_team_details',[]))
    p2t=(p2_lead.get('types') if isinstance(p2_lead,dict) else []) or _ttypes(bj.get('p2_team_details',[]))
    p1m=type_multiplier(p1t,p2t); p2m=type_multiplier(p2t,p1t)
    p1s_ = float(p1_lead.get('base_spe',0)) if isinstance(p1_lead,dict) else 0.0
    p2s_ = float(p2_lead.get('base_spe',0)) if isinstance(p2_lead,dict) else 0.0

    # Priors enrichments
    p2_infer = _infer_p2_from_priors(bj)
    p1_move_prior = _lead_move_type_priors(p1_lead if isinstance(p1_lead,dict) else {})
    p2_move_prior = _lead_move_type_priors(p2_lead if isinstance(p2_lead,dict) else {})

    # NEW: alive after 30-turn aggregates (computed in original POV, then swapped if needed)
    p1_alive = _alive_aggregates('p1', bj, max_turns=max_turns)
    p2_alive = _alive_aggregates('p2', bj, max_turns=max_turns)
    if not player_is_p1:
        p1_alive, p2_alive = p2_alive, p1_alive

    # Collect
    rec.update(p1s); rec.update(p2s)
    rec.update({f'p1_{k}':v for k,v in p1d.items()})
    rec.update({f'p2_{k}':v for k,v in p2d.items()})
    rec.update(p1_alive); rec.update(p2_alive)
    for k in list(p1s.keys()) + list({f'p1_{k}':v for k,v in p1d.items()}.keys()) + \
             [kk for kk in p1_alive.keys() if kk.startswith('p1_')]:
        rec[k.replace('p1_','diff_')] = rec.get(k,0) - rec.get(k.replace('p1_','p2_'),0)

    rec.update({
        'p1_type_off_mult':p1m,'p2_type_off_mult':p2m,'diff_type_off_mult':p1m-p2m,
        'p1_lead_base_spe':p1s_,'p2_lead_base_spe':p2s_,'diff_lead_base_spe':p1s_-p2s_,
        'p1_outspeed_prob': 1.0/(1.0+np.exp(-0.05*(p1s_-p2s_))),
        'diff_outspeed_prob': (1.0/(1.0+np.exp(-0.05*(p1s_-p2s_)))) - (1.0 - (1.0/(1.0+np.exp(-0.05*(p1s_-p2s_))))),
    })
    rec.update({f'p2_prior_{k}':v for k,v in p2_infer.items()})
    rec.update({f'p1_prior_{k}':v for k,v in p1_move_prior.items()})
    rec.update({f'p2_prior_{k}':v for k,v in p2_move_prior.items()})
    return rec

def create_feature_df(file_path, max_turns=30, show_progress: bool = True):
    rows=[]
    with open(file_path,'r',encoding='utf-8') as f:
        it = tqdm(f, desc=f"Building features from {getattr(file_path,'name',str(file_path))}", unit="battle") if show_progress else f
        for line in it:
            rows.append(process_battle(json.loads(line), max_turns=max_turns))
    df = pd.DataFrame(rows)
    if show_progress:
        tqdm.write(f"[features] Built dataframe: {df.shape[0]} rows, {df.shape[1]} cols")
    return df
