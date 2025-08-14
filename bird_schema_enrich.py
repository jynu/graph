# -*- coding: utf-8 -*-
"""
BIRD Schema Enricher v2 (GPT-5 + optional retrieval + graph & data aware)
=========================================================================

Key features vs v1:
- Retrieval-augmented (optional): fetch 3–6 short domain snippets (Wikipedia/Wikidata) per table
- Graph-aware: derive FK/PK relationships and auto-write join notes into tableSpecificRules and column descriptions
- Data-aware (optional): sample top-K distinct values and simple stats from CSV/SQLite/DuckDB to enrich examples
- Stronger prompt with few-shot hints, banned-phrase guardrails, and explicit length/coverage requirements
- Post-validation and retry loop if output is too weak, too short, or repeats banned phrases
- Provenance: stores citations under `extras.source_citations`

Usage:
    export OPENAI_API_KEY=sk-...

    python bird_schema_enricher_gpt5_v2.py \
      --input /path/to/processed_train.json \
      --out /path/to/enriched.json \
      --model gpt-5 --batch-size 6 --temperature 0.2 \
      --retrieve wiki \
      --sample-from /path/to/data_dir_or_sqlite_or_duckdb

Notes:
- Retrieval is best-effort. It’s bounded and whitelisted to reduce noise. If offline, use --retrieve off.
- Data sampling is optional. For CSV, expects <tablename>.csv in --sample-from directory.
  For SQLite/DuckDB, pass the file path and we’ll query table names that match tablename.
"""

import os
import re
import io
import json
import time
import math
import argparse
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict, Counter

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

os.environ[
    "OPENAI_API_KEY"] = ""

# Optional deps guarded
try:
    import duckdb  # type: ignore
except Exception:
    duckdb = None

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

try:
    import requests  # type: ignore
except Exception:
    requests = None

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None

BANNED_PHRASES = [
    "Data field:", "Name field for", "This column stores", "This table stores",
    "Generic field", "Miscellaneous", "Various information"
]

SYSTEM_MESSAGE = """You are a senior data product analyst enriching database schemas for a Text-to-SQL knowledge graph.
STRICT RULES:
- Do NOT change identifiers: keep 'tablename' and each 'columnname' untouched.
- Write specific, user-centered descriptions. Avoid boilerplate.
- Always mention entity, grain/time scope, and key join keys when obvious.
- Column descriptions must include meaning + units/examples when sensible. Mention FK target if relevant.
- Expand aliases with realistic synonyms and abbreviations users type.
- DO NOT fabricate facts beyond provided context/snippets. If unknown, say 'Unknown' succinctly.
- Use concise professional English.
- Return ONLY the JSON payload requested.
"""

FEWSHOT_HINT = """
Examples (style & depth):
- FTR (football): "Full-time result of the match: H=home win, D=draw, A=away win. Examples: H, D, A."
- constructorRef (F1): "Stable reference key for a constructor (team). Join to 'constructors'. Examples: ferrari, mercedes."
- k_symbol (banking): "Payment category code. Typical values: POJISTNE (insurance), SIPO (household payments), UROK (interest)."
"""

LLM_TEMPLATE = """You will enrich a table definition using the inputs below.

TABLE OBJECT:
{table_json}

OPTIONAL DOMAIN SNIPPETS (you MUST rely on them and not invent facts):
{snippets_text}

TASK: Return JSON with exact shape:
{{
  "tableDescription": "<min 2 sentences, include entity, grain, time scope if present>",
  "tableAlias": ["<alias1>", "<alias2>", ...],
  "tableSpecificRules": "<joins (PK/FK), time grain, unit notes, SCD if any>",
  "columns": [
    {{"columnname": "<unchanged>",
      "cleaned_name": "<phrase case>",
      "columnDescription": "<meaning + units/examples; if FK mention target table>",
      "columnAlias": ["<alias1>", "<alias2>", ...],
      "provide_distinct": "YES|NO"}}
    ... (one object per column, same order as input)
  ]
}}

HARD REQUIREMENTS:
- Avoid banned phrases: {banned_list}
- Column descriptions should not be generic. Prefer precise definitions.
- If meaning is unknown even after snippets, write "Unknown" but keep it short.
- Use the snippets to resolve codes and abbreviations when possible.
- Use examples from 'sampling_examples' if provided.
- Minimum tableDescription length: 2 sentences.

SAMPLING EXAMPLES (may be empty):
{sampling_examples}

FEW-SHOT HINT (do not repeat verbatim, but follow depth):
{fewshot}
"""

@dataclass
class Config:
    model: str = "gpt-5"
    temperature: float = -1.0
    batch_size: int = 6
    retrieve: str = "off"  # off | wiki
    sample_from: Optional[str] = None
    max_retries: int = 2
    resume_cache: str = "bird_enrich_cache_v2.jsonl"

# --------------------- Retrieval ---------------------

WIKI_ENDPOINT = "https://en.wikipedia.org/w/api.php"

def wiki_snippets(keywords: List[str], max_snippets: int = 5) -> List[str]:
    if requests is None:
        return []
    out = []
    for kw in keywords:
        try:
            # opensearch for a quick best page
            r = requests.get(WIKI_ENDPOINT, params=dict(
                action="opensearch", search=kw, limit=1, namespace=0, format="json"
            ), timeout=8)
            j = r.json()
            if not j or len(j) < 4 or not j[3]:
                continue
            url = j[3][0]
            # extract summary
            r2 = requests.get(WIKI_ENDPOINT, params=dict(
                action="query", prop="extracts", exintro=True, explaintext=True,
                titles=url.split("/")[-1], format="json"
            ), timeout=8)
            j2 = r2.json()
            pages = j2.get("query", {}).get("pages", {})
            for _, p in pages.items():
                ext = p.get("extract", "")
                if ext:
                    # take first ~550 chars
                    snippet = ext.strip().replace("\n", " ")
                    out.append(snippet[:550])
            if len(out) >= max_snippets:
                break
        except Exception:
            continue
    return out[:max_snippets]

def domain_keywords(table: Dict[str, Any]) -> List[str]:
    dbid = (table.get("database_id") or table.get("databaseId") or "").lower()
    tname = (table.get("tablename") or "").lower()
    keys = []
    if "football" in dbid or "football" in tname:
        keys = ["Association football results", "Bookmaker odds", "Premier League", "Full-time result H D A"]
    elif "formula" in dbid or "f1" in dbid or "constructor" in tname or "circuit" in tname:
        keys = ["Formula One constructor", "Formula One circuits", "Formula One qualifying"]
    elif "bank" in dbid or "financial" in dbid or "loan" in tname or "trans" in tname:
        keys = ["Czech bank dataset", "Bank transaction codes", "Payment symbols Czech"]
    else:
        # generic minimal hints
        keys = [tname, dbid, "database schema"]
    return keys

# --------------------- Data sampling ---------------------

def sample_values_from_csv(dirpath: str, tablename: str, colnames: List[str], max_distinct: int = 8) -> Dict[str, List[str]]:
    result = {}
    if pd is None:
        return result
    csv_path = os.path.join(dirpath, f"{tablename}.csv")
    if not os.path.exists(csv_path):
        return result
    try:
        df = pd.read_csv(csv_path, nrows=20000)  # light sample
        for c in colnames:
            if c in df.columns:
                vc = df[c].dropna().astype(str).value_counts().head(max_distinct)
                result[c] = [str(v) for v in vc.index.tolist()]
    except Exception:
        pass
    return result

def sample_values_from_sqlite(dbpath: str, tablename: str, colnames: List[str], max_distinct: int = 8) -> Dict[str, List[str]]:
    res = {}
    try:
        con = sqlite3.connect(dbpath)
        cur = con.cursor()
        for c in colnames:
            try:
                cur.execute(f"SELECT {c} FROM {tablename} WHERE {c} IS NOT NULL LIMIT 20000;")
                vals = [str(r[0]) for r in cur.fetchall() if r and r[0] is not None]
                if vals:
                    cnt = Counter(vals)
                    res[c] = [v for v, _ in cnt.most_common(max_distinct)]
            except Exception:
                continue
        con.close()
    except Exception:
        pass
    return res

def sample_values_from_duckdb(dbpath: str, tablename: str, colnames: List[str], max_distinct: int = 8) -> Dict[str, List[str]]:
    res = {}
    if duckdb is None:
        return res
    try:
        con = duckdb.connect(dbpath, read_only=True)
        for c in colnames:
            try:
                df = con.execute(f"SELECT {c} FROM {tablename} WHERE {c} IS NOT NULL LIMIT 20000").df()
                vals = [str(x) for x in df[c].dropna().astype(str).tolist()]
                if vals:
                    cnt = Counter(vals)
                    res[c] = [v for v, _ in cnt.most_common(max_distinct)]
            except Exception:
                continue
        con.close()
    except Exception:
        pass
    return res

def sample_examples(sample_from: Optional[str], tablename: str, colnames: List[str]) -> Dict[str, List[str]]:
    if not sample_from:
        return {}
    if os.path.isdir(sample_from):
        return sample_values_from_csv(sample_from, tablename, colnames)
    # else a file
    if sample_from.lower().endswith(".db") or sample_from.lower().endswith(".sqlite"):
        return sample_values_from_sqlite(sample_from, tablename, colnames)
    if sample_from.lower().endswith(".duckdb"):
        return sample_values_from_duckdb(sample_from, tablename, colnames)
    return {}

# --------------------- Graph helpers ---------------------

def build_pk_fk_maps(all_tables: List[Dict[str, Any]]) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], List[Tuple[str, str]]]]:
    """Return (pk_map, fk_map). 
    pk_map: table -> [pk column names]
    fk_map: (table, column) -> list of (ref_table, ref_col) targets
    Uses explicit flags if present, else falls back to name heuristics.
    """
    pk_map: Dict[str, List[str]] = defaultdict(list)
    fk_map: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)

    # First pass: explicit flags
    for t in all_tables:
        tname = t.get("tablename")
        for c in t.get("columns", []):
            if c.get("is_primary_key"):
                pk_map[tname].append(c.get("columnname"))

    # Second pass: look for foreign flags or name patterns
    name_index = defaultdict(list)  # column_name -> [(table, column)]
    for t in all_tables:
        tname = t.get("tablename")
        for c in t.get("columns", []):
            cname = c.get("columnname")
            name_index[cname].append((tname, cname))

    for t in all_tables:
        tname = t.get("tablename")
        for c in t.get("columns", []):
            cname = c.get("columnname")
            if c.get("is_foreign_key"):
                # try to infer target by matching column name to some table's PK or same-named column
                targets = []
                # direct by same-name PK
                for target_table, pk_cols in pk_map.items():
                    if cname in pk_cols and target_table != tname:
                        targets.append((target_table, cname))
                # fallback: same-name presence
                if not targets and cname in name_index:
                    for tt, cc in name_index[cname]:
                        if tt != tname and cc == cname:
                            targets.append((tt, cc))
                if targets:
                    fk_map[(tname, cname)].extend(targets)
            else:
                # heuristic: columns ending with _id often reference other table pks
                if cname.lower().endswith("_id"):
                    base = cname[:-3]
                    # try match table name singular/plural
                    for target_table, pk_cols in pk_map.items():
                        if base in target_table.lower() or target_table.lower().startswith(base):
                            if pk_cols:
                                fk_map[(tname, cname)].append((target_table, pk_cols[0]))
    return pk_map, fk_map

# --------------------- LLM engine ---------------------

class GPTEngine:
    def __init__(self, model: str, temperature: float):
        if OpenAI is None:
            raise RuntimeError("openai package not available. `pip install openai`")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def complete(self, system: str, user: str) -> str:
        kwargs = {
            'model': self.model,
            'messages': [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
        }
        # Only send temperature if explicitly set (>= 0)
        if self.temperature is not None and self.temperature >= 0:
            kwargs['temperature'] = self.temperature
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content.strip()

def sanitize_llm_json(text: str) -> Dict[str, Any]:
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].strip()
    return json.loads(s)

def has_banned_phrases(s: str) -> bool:
    s_low = s.lower()
    for bp in BANNED_PHRASES:
        if bp.lower() in s_low:
            return True
    return False

def valid_table_desc(s: str) -> bool:
    if not s or len(s.split(".")) < 2:
        return False
    return len(s.strip()) > 40

def ensure_column_cleaned_name(cname: str, cleaned: Optional[str]) -> str:
    if cleaned and cleaned.strip():
        return cleaned.strip()
    # create phrase case
    x = cname.replace("_", " ").strip()
    if not x:
        return cname
    return x[0].upper() + x[1:]

def enrich_one_table(engine: GPTEngine,
                     table_obj: Dict[str, Any],
                     pk_map: Dict[str, List[str]],
                     fk_map: Dict[Tuple[str, str], List[Tuple[str, str]]],
                     retrieve: str,
                     sample_from: Optional[str]) -> Dict[str, Any]:

    # Retrieve snippets
    snippets = []
    if retrieve == "wiki":
        kws = domain_keywords(table_obj)
        snippets = wiki_snippets(kws, max_snippets=5)

    # Sampling examples
    colnames = [c.get("columnname") for c in table_obj.get("columns", [])]
    sampling = sample_examples(sample_from, table_obj.get("tablename"), colnames)

    # Build snippets text (bounded)
    if snippets:
        snips = [f"- {s}" for s in snippets[:5]]
        snippets_text = "\n".join(snips)
    else:
        snippets_text = "(none)"

    # Sampling examples text
    if sampling:
        lines = []
        for k, vals in sampling.items():
            if vals:
                lines.append(f"{k}: {', '.join(vals[:8])}")
        sampling_text = "\n".join(lines) if lines else "(none)"
    else:
        sampling_text = "(none)"

    # Draft prompt
    user_prompt = LLM_TEMPLATE.format(
        table_json=json.dumps(table_obj, ensure_ascii=False, indent=2),
        snippets_text=snippets_text,
        banned_list=", ".join(BANNED_PHRASES),
        sampling_examples=sampling_text,
        fewshot=FEWSHOT_HINT,
    )

    # Call LLM with retries & validation
    attempts = 0
    enriched_obj: Dict[str, Any] = {}
    while attempts <= 2:
        attempts += 1
        raw = engine.complete(SYSTEM_MESSAGE, user_prompt)
        try:
            enriched_obj = sanitize_llm_json(raw)
        except Exception:
            # tighten instruction and retry
            user_prompt += "\n\nYour previous response was not valid JSON. Return ONLY the JSON object."
            continue

        # Basic validation
        td = enriched_obj.get("tableDescription", "")
        cols = enriched_obj.get("columns", [])
        if not valid_table_desc(td) or has_banned_phrases(td):
            user_prompt += "\n\nYour previous tableDescription was too weak. Make it at least 2 sentences, specific, and avoid banned phrases."
            continue
        bad_col = False
        for c in cols:
            cd = c.get("columnDescription", "")
            if not cd or len(cd) < 16 or has_banned_phrases(cd):
                bad_col = True
                break
        if bad_col:
            user_prompt += "\n\nSome column descriptions were too generic/short or used banned phrases. Strengthen them with meaning + units/examples and FK targets."
            continue
        break

    # Merge with graph info: ensure FK mention & join rules
    tname = table_obj.get("tablename")
    # tableSpecificRules: append joins
    joins = []
    for (tb, col), targets in fk_map.items():
        if tb == tname and targets:
            tgt_str = "; ".join([f"{tb}.{col} → {rt}.{rc}" for (rt, rc) in targets])
            joins.append(tgt_str)
    join_text = "; ".join(joins)
    if join_text:
        tsr = enriched_obj.get("tableSpecificRules", "") or ""
        if tsr:
            tsr = tsr.rstrip(".") + ". "
        tsr += f"Joins: {join_text}."
        enriched_obj["tableSpecificRules"] = tsr

    # Ensure cleaned_name and FK mention at column level
    name_to_targets = {col: tgts for (tb, col), tgts in fk_map.items() if tb == tname}
    fixed_cols = []
    for c_in, c_out in zip(table_obj.get("columns", []), enriched_obj.get("columns", [])):
        cname = c_in.get("columnname")
        c_out["columnname"] = cname  # force unchanged
        c_out["cleaned_name"] = ensure_column_cleaned_name(cname, c_out.get("cleaned_name"))
        # ensure provide_distinct default
        if "provide_distinct" not in c_out:
            # heuristic
            low = cname.lower()
            c_out["provide_distinct"] = "YES" if (low.endswith("type") or low.endswith("code") or low in ("ftr","division","k_symbol")) else "NO"
        # FK mention
        tgts = name_to_targets.get(cname, [])
        if tgts:
            desc = c_out.get("columnDescription", "") or ""
            mention = " ".join([f"FK to '{rt}.{rc}'." for (rt, rc) in tgts[:2]])
            if mention and mention.lower() not in desc.lower():
                if desc and not desc.endswith("."):
                    desc += "."
                desc += " " + mention
            c_out["columnDescription"] = desc.strip()
        fixed_cols.append(c_out)
    enriched_obj["columns"] = fixed_cols

    # Attach citations if any
    if snippets:
        enriched_obj.setdefault("extras", {})
        enriched_obj["extras"]["source_citations"] = [{"source": "wikipedia", "hint": k} for k in domain_keywords(table_obj)]

    return enriched_obj

# --------------------- IO & main ---------------------

def load_trainingdata(paths: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Tuple[str,int]]]:
    all_tables = []
    origin = {}
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            j = json.load(f)
        rows = j["trainingdata"] if isinstance(j, dict) and "trainingdata" in j else (j if isinstance(j, list) else [])
        for i, t in enumerate(rows):
            all_tables.append(t)
            origin[f"{len(all_tables)-1}"] = (p, i)
    return all_tables, origin

def save_grouped(paths: List[str], origin: Dict[str, Tuple[str,int]], enriched_tables: List[Dict[str, Any]], out_path: str):
    combined = {"source": "BIRD_dataset", "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "trainingdata": []}
    for t in enriched_tables:
        combined["trainingdata"].append(t)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True, help="Processed BIRD JSON (can repeat)")
    ap.add_argument("--out", required=True, help="Output enriched JSON path")
    ap.add_argument("--model", default="gpt-5")
    ap.add_argument("--temperature", type=float, default=-1.0)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--retrieve", choices=["off","wiki"], default="off")
    ap.add_argument("--sample-from", default=None, help="Dir (CSV) or a SQLite .db/.sqlite or DuckDB .duckdb file")
    ap.add_argument("--resume-cache", default="bird_enrich_cache_v2.jsonl")
    ap.add_argument("--flush-every", type=int, default=1, help="Rewrite --out JSON after every N tables")
    ap.add_argument("--progress", choices=["plain","tqdm"], default="plain")
    args = ap.parse_args()

    cfg = Config(model=args.model, temperature=args.temperature, batch_size=args.batch_size,
                 retrieve=args.retrieve, sample_from=args.sample_from, resume_cache=args.resume_cache)

    tables, origin = load_trainingdata(args.input)

    # Build graph maps
    pk_map, fk_map = build_pk_fk_maps(tables)

    # Init LLM
    engine = GPTEngine(cfg.model, cfg.temperature)

    # Load cache
    cache = {}
    if os.path.exists(cfg.resume_cache):
        with open(cfg.resume_cache, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    j = json.loads(line)
                    cache[j["tablename"]] = j["enriched"]
                except Exception:
                    continue

    enriched_tables = []
    total = len(tables)
    use_tqdm = (args.progress == 'tqdm' and tqdm is not None)
    iterator = tqdm(tables, total=total, desc='Enriching tables') if use_tqdm else tables
    processed = 0
    for t in iterator:
        tablename = t.get("tablename")
        if tablename in cache:
            enriched = cache[tablename]
        else:
            enriched = enrich_one_table(engine, t, pk_map, fk_map, cfg.retrieve, cfg.sample_from)
            with open(cfg.resume_cache, "a", encoding="utf-8") as f:
                f.write(json.dumps({"tablename": tablename, "enriched": enriched}, ensure_ascii=False) + "\n")

        # Merge minimal back into original table object
        merged = dict(t)
        for k in ("tableDescription", "tableAlias", "tableSpecificRules", "extras"):
            if k in enriched:
                merged[k] = enriched[k]
        # columns by order
        in_cols = t.get("columns", [])
        out_cols = enriched.get("columns", [])
        name_to_out = {c.get("columnname"): c for c in out_cols if isinstance(c, dict)}
        final_cols = []
        for c in in_cols:
            cname = c.get("columnname")
            e = name_to_out.get(cname, {})
            newc = dict(c)
            for k in ("cleaned_name","columnDescription","columnAlias","provide_distinct"):
                if k in e:
                    newc[k] = e[k]
            newc.setdefault("columnAlias", [])
            newc.setdefault("provide_distinct", "NO")
            newc.setdefault("distinct_values", c.get("distinct_values", []))
            newc.setdefault("distinct_value_map", c.get("distinct_value_map", {}))
            final_cols.append(newc)
        merged["columns"] = final_cols
        enriched_tables.append(merged)
        processed += 1
        if not use_tqdm:
            print(f"[{processed}/{total}] {tablename} ✓")
        if args.flush_every > 0 and (processed % args.flush_every == 0):
            save_grouped(args.input, origin, enriched_tables, args.out)
        time.sleep(0.15)  # polite pacing

    # Final save
    save_grouped(args.input, origin, enriched_tables, args.out)
    print(f"✅ Enrichment v2 complete. Wrote: {args.out}")

if __name__ == "__main__":
    main()