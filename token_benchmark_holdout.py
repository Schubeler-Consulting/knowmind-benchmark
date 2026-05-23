"""
Held-Out-Benchmark — der Lackmustest.

20 NEUE Fragen, die beim Pipeline-Tuning NICHT verwendet wurden.
Goldstandard sind Memory-Dateinamen (ohne .md) — exakter Match in den
Top-K-IDs der jeweiligen Methode.

Methoden, Token-Zählung, Preise identisch zu token_benchmark.py.
Wenn knowmind hier nicht standhält, war der Erfolg im ersten Test
nur Tuning-Bias.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

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

# Held-out: 20 Fragen die im ersten Benchmark NICHT vorkamen.
# Goldstandard = mindestens einer dieser Dateinamen (Slug ohne .md)
# muss in den Top-K-IDs vorkommen.
QUESTIONS: list[dict] = [
    {"q": "Wo läuft die OKR-App?", "gold": ["reference_okr_app", "project_okr_app", "feedback_okr"]},
    {"q": "Was ist DocTwin?", "gold": ["project_doctwin_status_2026-05-12", "reference_doctwin"]},
    {"q": "Welche Brand-Marke hat das Bettenparadies?", "gold": ["reference_bettenparadies_ci", "feedback_schreibweise_betten_paradies", "feedback_bpp_premium_voice"]},
    {"q": "Welche Telegram-Chat-ID hat Johann?", "gold": ["reference_telegram_chat_id", "feedback_telegram_offline"]},
    {"q": "Wie funktioniert der Karpathy-Loop?", "gold": ["project_sc_memory_mcp", "reference_karpathy_loop"]},
    {"q": "Was ist Rotando Booking?", "gold": ["reference_container_naming_ops", "feedback_rotando_deploy_checklist", "feedback_rotando_mandanten_tz"]},
    {"q": "Welche Lizenzmodell hat das ai-exzellenz-starter Repo?", "gold": ["project_ai_exzellenz_starter"]},
    {"q": "Wann gab es einen RAM-Leak im Memory-Service?", "gold": ["project_soviq_memory_leak"]},
    {"q": "Was wurde am Spec-Full-Rollout gemacht?", "gold": ["project_spec_full_rollout_2026-05-13"]},
    {"q": "Welche Service-Token-Auth wird zwischen Control-Plane und Memory genutzt?", "gold": ["session_handoff_2026-05-12_knowmind_sprint2"]},
    {"q": "Wie schreibt man Em-Dashes in SC-Texten?", "gold": ["feedback_kein_ki_geschlabber", "feedback_keine_antithesen"]},
    {"q": "Was steht in der Blogpost-Featured-Image-Vorlage?", "gold": ["feedback_blogbeitragsbilder_via_codex"]},
    {"q": "Wer ist Hanna?", "gold": ["project_sc_memory_deploy_bettenparadies", "reference_hanna"]},
    {"q": "Welche IP hat schuebeler.de?", "gold": ["reference_schuebeler_de_hosting"]},
    {"q": "Was ist das Knowmind-Pre-Ingest?", "gold": ["reference_knowmind_preference_extractor"]},
    {"q": "Wann wurde das Vater-Buch v6 fertig?", "gold": ["session_handoff_2026-05-22_knowmind_install"]},
    {"q": "Welche Migrations-Schritte gab es beim Knowmind-Bugfix-Sprint?", "gold": ["project_knowmind_bugfix_sprint_2026-05-22"]},
    {"q": "Wie viele Memories liegen im Knowmind-Mandanten?", "gold": ["project_knowmind_audit_2026-05-22", "reference_sc_memory_ist_knowmind"]},
    {"q": "Wie deploye ich auf die Knowmind-Plattform?", "gold": ["reference_knowmind_platform_deploy"]},
    {"q": "Was ist der KI-Start-Kompass?", "gold": ["project_ki_beratung_nrw_funnel"]},
]


def count_tokens(s: str) -> int:
    return len(ENC.encode(s))


def _slug_norm(s: str) -> str:
    """Underscore und Bindestrich werden gleichgesetzt — beide Varianten
    referenzieren denselben Memory-Slug. Das war ein Bug im ursprünglichen
    `hit`: Memories werden teils mit Dashes (Postgres-Title-Normalisierung)
    und teils mit Underscores (Original-Filename) gespeichert, der Test
    matcht sonst nur eine der beiden Varianten."""
    return s.lower().replace("_", "-")


def hit(ids: list[str], gold: list[str], excerpts: list[str]) -> bool:
    """Treffer wenn mindestens ein gold-Slug in den IDs oder im
    File-Content der Top-K vorkommt (case-insensitive, dash/underscore-
    normalisiert)."""
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
        timeout=20,
    )
    if r.status_code != 200:
        return {}
    j = r.json()
    try:
        return json.loads(j["result"]["content"][0]["text"])
    except (KeyError, IndexError, json.JSONDecodeError):
        return j.get("result", {})


def _hit_idents(h: dict) -> list[str]:
    """Sammelt alle Identifikatoren eines Knowmind-Hits, gegen die der
    Goldstandard-Slug-Substring matchen darf: source_id (Slug oder UUID),
    title (oft Slug mit Bindestrich), name (Memory-Knoten-Name), chunk_id.
    Vector-Chunks haben oft eine UUID als source_id und den Slug nur im
    title — ohne title sind sie für Slug-Match unsichtbar."""
    md = h.get("metadata", {}) or {}
    return [
        str(md.get("source_id", "")),
        str(md.get("title", "")),
        str(md.get("name", "")),
        str(h.get("chunk_id", "")),
    ]


def method_vector_only(q: str, k: int = K):
    t0 = time.perf_counter()
    data = _kmd(q, k * 4)
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
    print("HELD-OUT-Benchmark — 20 vorher unbenutzte Fragen, schärferer Goldstandard")
    print(f"k={K} · Pipeline-Filter: NUR die unstrittigen 7 Müll-Typen ausgeschlossen")
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
            mark = "✓" if h else "✗"
            print(f"        {name:>13s}: {mark} {tok:>6,}t  {lat:>5.0f}ms  top: {ids[:2]}".replace(",", "."))
        detailed.append(row)

    print("\n" + "=" * 92)
    print("Aggregat — HELD OUT")
    print("=" * 92)
    n = len(QUESTIONS)
    print(f"{'Methode':<15} {'Recall@5':>10} {'Ø Tokens':>12} {'Ø Latenz':>10} {'Ø Kosten':>14}")
    print("-" * 92)
    summary = {}
    for name, _ in METHODS:
        m = per[name]
        recall = m["hits"] / n * 100
        avg_t = sum(m["tokens"]) / len(m["tokens"]) if m["tokens"] else 0
        avg_l = sum(m["latency"]) / len(m["latency"]) if m["latency"] else 0
        cost_ct = avg_t / 1_000_000 * PRICE_OPUS_INPUT_PER_MTOK * USD_TO_EUR * 100
        print(f"{name:<15} {recall:>9.1f}% {avg_t:>12,.0f} {avg_l:>9.0f}ms {cost_ct:>10.3f} ct".replace(",", "."))
        summary[name] = {"recall_at_5_pct": round(recall, 1), "avg_tokens": round(avg_t),
                          "avg_latency_ms": round(avg_l), "avg_cost_eur_ct": round(cost_ct, 3)}

    out = Path(__file__).parent / "token_benchmark_holdout_result.json"
    out.write_text(json.dumps({"per_method": summary, "detailed": detailed},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRoh-Daten: {out}")


if __name__ == "__main__":
    main()
