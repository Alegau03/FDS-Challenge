# Build & load corpus-level priors WITH PROGRESS.
import json
from collections import defaultdict, Counter
from typing import Dict, Any, Iterable
from pathlib import Path
from tqdm import tqdm
import config

def _add_counter(d: Dict[str, Counter], key: str, items: Iterable[str]):
    c = d.setdefault(key, Counter())
    c.update([x for x in items if x and x != key])

def build_priors(train_path: Path = config.TRAIN_JSONL, out_path: Path = config.PRIORS_PATH):
    cooc = defaultdict(Counter)
    move_type = defaultdict(Counter)
    base_stats = defaultdict(lambda: Counter({"cnt":0,"base_hp":0,"base_atk":0,"base_def":0,"base_spa":0,"base_spd":0,"base_spe":0}))

    def _accumulate_team(team):
        names = [ (p.get("name") or "").lower() for p in (team or []) if p ]
        for p in (team or []):
            nm = (p.get("name") or "").lower()
            if not nm: continue
            _add_counter(cooc, nm, names)
            bs = base_stats[nm]; bs["cnt"] += 1
            for k in ["base_hp","base_atk","base_def","base_spa","base_spd","base_spe"]:
                bs[k] += float(p.get(k,0))

    with open(train_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Scanning priors from {getattr(train_path,'name',str(train_path))}", unit="battle"):
            b = json.loads(line)
            p1_team = b.get("p1_team_details", []) or []
            p2_team = b.get("p2_team_details") or []
            if not p2_team:
                lead = b.get("p2_lead_details") or {}
                p2_team = [lead] if lead else []
            _accumulate_team(p1_team); _accumulate_team(p2_team)

            for t in b.get("battle_timeline", b.get("battle_log", [])) or []:
                for side in ["p1","p2"]:
                    st = (t.get(f"{side}_pokemon_state") or {})
                    mv = (t.get(f"{side}_move_details") or {})
                    name = (st.get("name") or "").lower()
                    if not name: continue
                    if mv:
                        typ = (mv.get("type") or "").lower()
                        if typ: move_type[name][typ] += 1

    cooc_prob = {mon:{k:v/sum(cnt.values()) for k,v in cnt.items()} for mon,cnt in cooc.items() if sum(cnt.values())>0}
    move_type_prob = {mon:{k:v/sum(cnt.values()) for k,v in cnt.items()} for mon,cnt in move_type.items() if sum(cnt.values())>0}
    base_stats_mean = {mon:{k: float(c[k])/(c["cnt"] or 1) for k in ["base_hp","base_atk","base_def","base_spa","base_spd","base_spe"]} for mon,c in base_stats.items()}

    priors = {"cooc_prob": cooc_prob, "move_type_prob": move_type_prob, "base_stats_mean": base_stats_mean, "meta": {"built_from": str(train_path)}}
    out_path.write_text(json.dumps(priors), encoding="utf-8")
    tqdm.write(f"[priors] Saved to {out_path}")
    return priors

def load_priors(path: Path = config.PRIORS_PATH) -> Dict[str, Any]:
    if path.exists():
        tqdm.write(f"[priors] Loaded {path}")
        return json.loads(path.read_text(encoding="utf-8"))
    return build_priors()

if __name__ == "__main__":
    build_priors()
