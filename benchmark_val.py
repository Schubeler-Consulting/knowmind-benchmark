"""
Validation-Benchmark — 30 distinct questions, disjoint from train + held-out.

Goldstandard = mindestens einer der gold-Slugs muss in den Top-K-IDs
oder Excerpts der Methode auftauchen (case-insensitive Substring).

Wird im Karpathy-Loop nach jeder Pipeline-Änderung gemessen, um Overfit
gegen das Train-Set zu erkennen. Held-Out wird hier NICHT angefasst.

Topics gewählt aus dem Korpus mit klarer Quellen-Datei, ohne die in
token_benchmark.py oder token_benchmark_holdout.py vorkommen.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import os
import requests
import tiktoken

PRICE_OPUS_INPUT_PER_MTOK = 15.0
USD_TO_EUR = 0.92
ENC = tiktoken.get_encoding("cl100k_base")
MEMORY_DIR = Path(os.environ.get("KNOWMIND_BENCHMARK_CORPUS", ".")).expanduser()
KNOWMIND_API_URL = os.environ.get("KNOWMIND_API_URL", "https://knowmind.de")
KNOWMIND_TOKEN = os.environ.get("KNOWMIND_TOKEN", "")
TENANT_ID = os.environ.get("KNOWMIND_TENANT_ID", "")
if not KNOWMIND_TOKEN:
    raise SystemExit("KNOWMIND_TOKEN nicht gesetzt — siehe README.")
K = 5

# Validation-Set: 30 Fragen, deren Themen weder im Train- noch im
# Held-Out-Set vorkommen. Hand-kuratiert aus den Memory-Files.
QUESTIONS: list[dict] = [
    {"q": "Welche IP hat der kleine Hetzner-Server?", "gold": ["reference_hetzner_klein_detail", "feedback_hetzner_klein_rotando"]},
    {"q": "Was steht in der Brevo-Reference?", "gold": ["reference_brevo"]},
    {"q": "Wie ist das LanguageTool eingebunden?", "gold": ["reference_languagetool_stage"]},
    {"q": "Wie sieht das Auth.js v5 Edge-Split-Pattern aus?", "gold": ["reference_authjs_v5_edge_split"]},
    {"q": "Was ist der Chunk-Embedding-Layer?", "gold": ["reference_chunk_embedding_layer"]},
    {"q": "Welche Domains gehören zur KI-Marke?", "gold": ["reference_domains_ki"]},
    {"q": "Wie ist Flux2 aufgesetzt?", "gold": ["reference_flux2_setup"]},
    {"q": "Welche Hardware hat der Dev-Rechner?", "gold": ["reference_dev_rechner_hardware"]},
    {"q": "Was ist Hilde-Tunnel zum LM Studio?", "gold": ["reference_hilde_tunnel_lm_studio"]},
    {"q": "Welche Lizenzen sind im Abo?", "gold": ["reference_lizenzen_abo"]},
    {"q": "Welche Erstberatung-Booking-URL nutzen wir?", "gold": ["reference_erstberatung_booking", "reference_bookings_urls_themen"]},
    {"q": "Was ist im MS365 MCP enthalten?", "gold": ["reference_ms365_mcp"]},
    {"q": "Wie ist n8n eingebunden?", "gold": ["reference_n8n"]},
    {"q": "Welche Partner-CCMS gibt es?", "gold": ["reference_partner_icms"]},
    {"q": "Was steht in der Stripe-Reference?", "gold": ["reference_stripe", "reference_knowmind_stripe"]},
    {"q": "Wie funktioniert Tailscale für LM Studio?", "gold": ["reference_tailscale_lm_studio", "reference_tailscale_bridge"]},
    {"q": "Wie startet man den Telegram-Bot?", "gold": ["reference_telegram_start"]},
    {"q": "Wie ist der Vater-Buch-Edits-Workflow?", "gold": ["reference_vater_buch_edits_workflow"]},
    {"q": "Welche WP-REST Redakteur-Endpoints gibt es?", "gold": ["reference_wp_rest_redakteur"]},
    {"q": "Welche Whitepaper hat SC veröffentlicht?", "gold": ["reference_sc_whitepapers"]},
    {"q": "Was ist das SMA-Joblevel-Tool?", "gold": ["reference_sma_joblevel_tool"]},
    {"q": "Wie ist das Brain-Remodel vom 4. Mai aufgebaut?", "gold": ["reference_brain_remodel_2026-05-04"]},
    {"q": "Wie kommt man an das CodeBurn-Dashboard?", "gold": ["reference_codeburn_dashboard"]},
    {"q": "Was beschreibt der Master-Plan Orchestrator?", "gold": ["project_master_plan_orchestrator"]},
    {"q": "Wie ist die LMS-Agent-Engine-Architektur?", "gold": ["project_lms_agent_engine_architecture", "project_lms_agent_engine_archi"]},
    {"q": "Was ist project_buecherhaus_testmandant?", "gold": ["project_buecherhaus_testmandant_2026-05-14"]},
    {"q": "Wie funktioniert der OpenProject-Stack?", "gold": ["project_openproject"]},
    {"q": "Was beschreibt der LMS-Roadmap-Plan?", "gold": ["project_lms_roadmap"]},
    {"q": "Was steht im wp_learning_detail?", "gold": ["project_wp_learning_detail"]},
    {"q": "Was beschreibt die SC-Calendar-Mirror-Datei?", "gold": ["project_sc_calendar_mirror"]},
]


def count_tokens(s: str) -> int:
    return len(ENC.encode(s))


def _slug_norm(s: str) -> str:
    """Slug-Match-Normalisierung: dash/underscore werden gleichgesetzt."""
    return s.lower().replace("_", "-")


def hit(ids: list[str], gold: list[str], excerpts: list[str]) -> bool:
    blob = _slug_norm(" ".join(ids + excerpts))
    return any(_slug_norm(g) in blob for g in gold)


def method_ripgrep(q: str, k: int = K):
    t0 = time.perf_counter()
    tokens = [t for t in re.findall(r"\w+", q.lower()) if len(t) >= 3]
    counts: dict[str, int] = {}
    for f in MEMORY_DIR.glob("*.md"):
        try:
            text = f.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue
        c = sum(text.count(t) for t in tokens)
        if c > 0:
            counts[str(f)] = c
    ranked = sorted(counts.items(), key=lambda x: -x[1])[:k]
    ids = [Path(p).stem for p, _ in ranked]
    excerpts = [Path(p).read_text(encoding="utf-8", errors="ignore")[:2000] for p, _ in ranked]
    return ids, excerpts, (time.perf_counter() - t0) * 1000


def method_fname(q: str, k: int = K):
    t0 = time.perf_counter()
    tokens = [t for t in re.findall(r"\w+", q.lower()) if len(t) >= 3]
    counts: dict[str, int] = {}
    for f in MEMORY_DIR.glob("*.md"):
        name = f.stem.lower()
        c = sum(1 for t in tokens if t in name)
        if c > 0:
            counts[str(f)] = c
    ranked = sorted(counts.items(), key=lambda x: -x[1])[:k]
    ids = [Path(p).stem for p, _ in ranked]
    excerpts = [Path(p).read_text(encoding="utf-8", errors="ignore")[:2000] for p, _ in ranked]
    return ids, excerpts, (time.perf_counter() - t0) * 1000


def _kmd(q: str, k: int):
    r = requests.post(
        f"{KNOWMIND_API_URL}/api/mcp/v1",
        headers={"Authorization": f"Bearer {KNOWMIND_TOKEN}", "Content-Type": "application/json"},
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
              "params": {"name": "knowmind.recall", "arguments": {"query": q, "k": k}}},
        timeout=30,
    )
    if r.status_code != 200:
        return {}
    j = r.json()
    try:
        return json.loads(j["result"]["content"][0]["text"])
    except (KeyError, IndexError, json.JSONDecodeError):
        return j.get("result", {})


def _hit_idents(h: dict) -> list[str]:
    """Alle Identifier eines Knowmind-Hits (source_id, title, name, chunk_id)."""
    md = h.get("metadata", {}) or {}
    return [
        str(md.get("source_id", "")),
        str(md.get("title", "")),
        str(md.get("name", "")),
        str(h.get("chunk_id", "")),
    ]


def method_vector_only(q: str, k: int = K):
    t0 = time.perf_counter()
    data = _kmd(q, min(k * 4, 25))
    latency = (time.perf_counter() - t0) * 1000
    hits = [h for h in data.get("hits", []) if h.get("source") == "vector"][:k]
    ids = [s for h in hits for s in _hit_idents(h) if s]
    excerpts = [str(h.get("content", ""))[:2000] for h in hits]
    return ids, excerpts, latency


def method_knowmind(q: str, k: int = K):
    t0 = time.perf_counter()
    data = _kmd(q, k)
    latency = (time.perf_counter() - t0) * 1000
    hits = data.get("hits", [])[:k]
    ids = [s for h in hits for s in _hit_idents(h) if s]
    excerpts = [str(h.get("content", ""))[:2000] for h in hits]
    return ids, excerpts, latency


METHODS = [("ripgrep", method_ripgrep), ("fname", method_fname),
           ("vector_only", method_vector_only), ("knowmind", method_knowmind)]


def main():
    print("=" * 92)
    print("VAL-Benchmark — 30 disjunkte Fragen (kein Overlap mit Train + Held-Out)")
    print(f"k={K}")
    print("=" * 92)

    per = {m: {"hits": 0, "tokens": [], "latency": []} for m, _ in METHODS}
    detailed = []

    for i, item in enumerate(QUESTIONS, 1):
        q, gold = item["q"], item["gold"]
        print(f"\n[{i:2d}/{len(QUESTIONS)}] {q}")
        print(f"        gold: {gold}")
        row = {"q": q, "gold": gold, "methods": {}}
        for name, fn in METHODS:
            try:
                ids, excerpts, lat = fn(q, K)
            except Exception as e:
                print(f"        {name:>13s}: FEHLER {e}")
                continue
            ctx = "\n\n---\n\n".join(excerpts)
            tok = count_tokens(ctx)
            h = hit(ids, gold, excerpts)
            per[name]["hits"] += 1 if h else 0
            per[name]["tokens"].append(tok)
            per[name]["latency"].append(lat)
            row["methods"][name] = {"hit": h, "tokens": tok, "latency_ms": round(lat), "top_ids": ids[:3]}
            mark = "OK" if h else "--"
            print(f"        {name:>13s}: {mark} {tok:>6,}t  {lat:>5.0f}ms  top: {ids[:2]}".replace(",", "."))
        detailed.append(row)

    print("\n" + "=" * 92)
    print("Aggregat — VAL")
    print("=" * 92)
    n = len(QUESTIONS)
    print(f"{'Methode':<15} {'Recall@5':>10} {'Ø Tokens':>12} {'Ø Latenz':>10}")
    print("-" * 92)
    summary = {}
    for name, _ in METHODS:
        m = per[name]
        recall = m["hits"] / n * 100
        avg_t = sum(m["tokens"]) / len(m["tokens"]) if m["tokens"] else 0
        avg_l = sum(m["latency"]) / len(m["latency"]) if m["latency"] else 0
        print(f"{name:<15} {recall:>9.1f}% {avg_t:>12,.0f} {avg_l:>9.0f}ms".replace(",", "."))
        summary[name] = {"recall_at_5_pct": round(recall, 1), "avg_tokens": round(avg_t),
                          "avg_latency_ms": round(avg_l)}

    out = Path(__file__).parent / "benchmark_val_result.json"
    out.write_text(json.dumps({"per_method": summary, "detailed": detailed},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRoh-Daten: {out}")


if __name__ == "__main__":
    main()
