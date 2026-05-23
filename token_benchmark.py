"""
Ehrlicher Retrieval-Benchmark: knowmind vs. realistische Alternativen.

Methoden im Vergleich (das sind die Tools, die ein Kunde OHNE knowmind
tatsächlich einsetzt, um seinen Memory-Korpus für eine LLM-Anfrage
aufzubereiten):

  A) ripgrep         — Volltextsuche, Top-K nach Match-Count
  B) Dateiname-Match — Naive "ich weiß welche Datei das ist"-Suche
  C) Vector-Only     — Pure Embedding-Search (klassisches RAG)
  D) knowmind        — Hybrid BM25 + Vector + Graph + Identity + Relationship

Gemessen pro Methode:
  - Recall@5: Lag das erwartete Quell-Memory in den Top-5?
  - Kontext-Tokens: Wie viele Tokens müsste der Kunde an Opus 4.7 schicken,
    wenn er die Top-5-Treffer als Kontext einbettet?
  - Latenz: Wie lange braucht die Methode?
  - Kosten: Input-Token × Opus-4.7-Preis

Goldstandard ist pro Frage hinterlegt: erwarteter Memory-Slug-Substring.
Voll-Dump als Strohmann ist BEWUSST raus.
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
PRICE_OPUS_OUTPUT_PER_MTOK = 75.0
USD_TO_EUR = 0.92

ENC = tiktoken.get_encoding("cl100k_base")
MEMORY_DIR = Path(os.environ.get("KNOWMIND_BENCHMARK_CORPUS", ".")).expanduser()

KNOWMIND_API_URL = os.environ.get("KNOWMIND_API_URL", "https://knowmind.de")
KNOWMIND_TOKEN = os.environ.get("KNOWMIND_TOKEN", "")
TENANT_ID = os.environ.get("KNOWMIND_TENANT_ID", "")
if not KNOWMIND_TOKEN:
    raise SystemExit(
        "KNOWMIND_TOKEN nicht gesetzt. Token aus knowmind.de/dashboard/tokens holen,\n"
        "  export KNOWMIND_TOKEN=kmt_...\n"
        "  export KNOWMIND_TENANT_ID=<UUID aus Dashboard>"
    )

K = 5

# Realistische SC-Fragen mit GOLDSTANDARD-Quelle.
# expected_slug = Substring der Datei/des Knotens, in dem die Antwort steht.
QUESTIONS: list[dict] = [
    {"q": "Wer leitet das Bettenparadies?", "expected": ["bettenparadies", "cordt-ulrich", "carl-august", "schuebeler_vater_familie"]},
    {"q": "Was ist SC?", "expected": ["schuebeler-consulting", "schuebeler_consulting"]},
    {"q": "Welche Tarife hat knowmind?", "expected": ["knowmind", "stripe", "pricing", "tier"]},
    {"q": "Auf welchem Server läuft das LMS?", "expected": ["hetzner", "lms", "wp-learning"]},
    {"q": "Wer ist Carl-August Schübeler?", "expected": ["carl-august", "schuebeler_vater_familie"]},
    {"q": "Wie ist die SSH-Verbindung zum Hetzner-Server?", "expected": ["hetzner", "ssh"]},
    {"q": "Welche Sprache verwenden wir für die Tech-Doku-Beratung?", "expected": ["deutsch", "sprache", "umlaut", "voice"]},
    {"q": "Wie heißt das WordPress-Theme der SC-Website?", "expected": ["consulting_page", "consulting-page", "sc_theme", "schuebeler-consulting"]},
    {"q": "Wann läuft das 7-Tage-Limit von Claude Pro zurück?", "expected": ["claude_7d", "reset"]},
    {"q": "Welche Edge-Typen gibt es im Wissensgraph?", "expected": ["edge", "graph", "schema", "sc_memory_graph"]},
    {"q": "Wie kann ich knowmind auf einem neuen Rechner einrichten?", "expected": ["knowmind", "install", "cli", "setup"]},
    {"q": "Was sind die wichtigsten SC-Werteversprechen?", "expected": ["sc_website", "ki_beratung", "marketing"]},
    {"q": "Welche MCP-Server sind in claude.json registriert?", "expected": ["mcp", "claude_json", "claude.json"]},
    {"q": "Wie lautet die Pricing-Strategie für Plus-Tier?", "expected": ["knowmind", "pricing", "plus", "stripe"]},
    {"q": "Welche Standorte hat die SC-Beratung?", "expected": ["beverungen", "owl", "weserbergland", "standort"]},
    {"q": "Was passiert mit Sessions wenn das LLM Kontext verliert?", "expected": ["memory", "context", "session"]},
    {"q": "Wie funktioniert die Memory-First-Standing-Order?", "expected": ["memory_first", "memory-first"]},
    {"q": "Was ist der Unterschied zwischen Plus und Pro?", "expected": ["knowmind", "pricing", "plus", "pro"]},
    {"q": "Wer hat den Bettenparadies-CI festgelegt?", "expected": ["bettenparadies_ci", "ci"]},
    {"q": "Wie kann ich Beziehungen im Graph manuell pflegen?", "expected": ["graph", "beziehung", "manual"]},
]


def count_tokens(s: str) -> int:
    return len(ENC.encode(s))


def _slug_norm(s: str) -> str:
    """Slug-Match-Normalisierung: dash/underscore werden gleichgesetzt."""
    return s.lower().replace("_", "-")


def hit_in_top_k(hits_text: list[str], expected: list[str]) -> bool:
    """True wenn mindestens ein expected-Substring in einem der hits vorkommt.
    Slug-Match ist dash/underscore-normalisiert — Memory-Titel können in
    beiden Schreibweisen abgelegt sein."""
    blob = _slug_norm(" ".join(hits_text))
    return any(_slug_norm(e) in blob for e in expected)


# ============================================================
# Methode A: ripgrep
# ============================================================
def method_ripgrep(query: str, k: int = K) -> tuple[list[str], list[str], float]:
    """Volltextsuche per ripgrep. Zählt Matches pro File, sortiert absteigend.
    Liefert (file_ids, file_contents_excerpted, latency_ms)."""
    t0 = time.perf_counter()
    # Splits Query in 2+-Buchstaben-Tokens, dedup
    tokens = list({t for t in re.findall(r"\w+", query.lower()) if len(t) >= 3})
    if not tokens:
        return [], [], 0.0
    # Wir nutzen ripgrep wenn verfügbar, sonst Python-Fallback
    try:
        # Alle Files durchscannen, Match-Count pro File
        counts: dict[str, int] = {}
        for tok in tokens:
            r = subprocess.run(
                ["rg", "-i", "-c", tok, str(MEMORY_DIR)],
                capture_output=True, text=True, timeout=20,
                encoding="utf-8", errors="ignore",
            )
            for line in r.stdout.splitlines():
                if ":" not in line:
                    continue
                path, n = line.rsplit(":", 1)
                try:
                    counts[path] = counts.get(path, 0) + int(n)
                except ValueError:
                    continue
        if not counts:
            return [], [], (time.perf_counter() - t0) * 1000
        ranked = sorted(counts.items(), key=lambda x: -x[1])[:k]
        ids = [Path(p).stem for p, _ in ranked]
        excerpts = []
        for p, _ in ranked:
            try:
                excerpts.append(Path(p).read_text(encoding="utf-8", errors="ignore")[:2000])
            except OSError:
                continue
        return ids, excerpts, (time.perf_counter() - t0) * 1000
    except FileNotFoundError:
        # Fallback ohne ripgrep
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


# ============================================================
# Methode B: Dateiname-Match
# ============================================================
def method_fname(query: str, k: int = K) -> tuple[list[str], list[str], float]:
    """Naive Dateiname-Suche: zählt Token-Treffer im Dateinamen."""
    t0 = time.perf_counter()
    tokens = list({t for t in re.findall(r"\w+", query.lower()) if len(t) >= 3})
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


# ============================================================
# Methode C: Vector-Only via Knowmind-API (Stages deaktiviert ausser vector)
# Hinweis: Die Memory-Service-API erlaubt aktuell kein selektives
# Stage-Disable. Wir simulieren das, indem wir die Recall-Antwort nehmen
# und nur die Hits mit source='vector' verwenden.
# ============================================================
def _knowmind_recall(query: str, k: int = K, k_overshoot: int = 1) -> dict:
    """JSON-RPC-Call gegen den MCP-Endpoint."""
    r = requests.post(
        f"{KNOWMIND_API_URL}/api/mcp/v1",
        headers={
            "Authorization": f"Bearer {KNOWMIND_TOKEN}",
            "Content-Type": "application/json",
        },
        json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "knowmind.recall", "arguments": {"query": query, "k": k * k_overshoot}},
        },
        timeout=20,
    )
    if r.status_code != 200:
        return {}
    j = r.json()
    # MCP-Response: result.content[0].text ist JSON-Payload
    try:
        content = j["result"]["content"][0]["text"]
        return json.loads(content)
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


def method_vector_only(query: str, k: int = K) -> tuple[list[str], list[str], float]:
    """Pure Vector-Hits aus der Recall-Antwort, alle anderen Stages ignoriert."""
    t0 = time.perf_counter()
    data = _knowmind_recall(query, k=k, k_overshoot=4)
    latency_ms = (time.perf_counter() - t0) * 1000
    hits = data.get("hits", [])
    vector_hits = [h for h in hits if h.get("source") == "vector"][:k]
    ids = [s for h in vector_hits for s in _hit_idents(h) if s]
    excerpts = [str(h.get("content", ""))[:2000] for h in vector_hits]
    return ids, excerpts, latency_ms


def method_knowmind(query: str, k: int = K) -> tuple[list[str], list[str], float]:
    """knowmind Hybrid — alle Stages aktiv."""
    t0 = time.perf_counter()
    data = _knowmind_recall(query, k=k)
    latency_ms = (time.perf_counter() - t0) * 1000
    hits = data.get("hits", [])[:k]
    ids = [s for h in hits for s in _hit_idents(h) if s]
    excerpts = [str(h.get("content", ""))[:2000] for h in hits]
    return ids, excerpts, latency_ms


# ============================================================
# Runner
# ============================================================
METHODS = [
    ("ripgrep", method_ripgrep),
    ("fname", method_fname),
    ("vector_only", method_vector_only),
    ("knowmind", method_knowmind),
]


def cost_input_usd(tokens: int) -> float:
    return tokens / 1_000_000 * PRICE_OPUS_INPUT_PER_MTOK


def main() -> None:
    print("=" * 88)
    print("Retrieval-Benchmark — knowmind vs. realistische Alternativen")
    print(f"Korpus: {len(list(MEMORY_DIR.glob('*.md')))} .md-Dateien aus {MEMORY_DIR}")
    print(f"Tenant: sc-internal · k={K} · Modell-Preis Opus 4.7: "
          f"{PRICE_OPUS_INPUT_PER_MTOK}/M In")
    print("=" * 88)

    per_method: dict[str, dict] = {m: {"hits": 0, "tokens": [], "latency": []} for m, _ in METHODS}
    detailed = []

    for i, item in enumerate(QUESTIONS, 1):
        q = item["q"]
        expected = item["expected"]
        print(f"\n[{i:2d}/{len(QUESTIONS)}] {q}")
        print(f"        expected: {expected}")
        row = {"q": q, "expected": expected, "methods": {}}
        for name, fn in METHODS:
            try:
                ids, excerpts, latency = fn(q, K)
            except Exception as e:  # noqa: BLE001
                print(f"        {name:>13s}: FEHLER {e}")
                continue
            ctx = "\n\n---\n\n".join(excerpts)
            tokens = count_tokens(ctx)
            hit = hit_in_top_k(ids + excerpts, expected)
            per_method[name]["hits"] += 1 if hit else 0
            per_method[name]["tokens"].append(tokens)
            per_method[name]["latency"].append(latency)
            row["methods"][name] = {
                "hit": hit, "tokens": tokens, "latency_ms": round(latency),
                "top_ids": ids[:3],
            }
            mark = "✓" if hit else "✗"
            print(f"        {name:>13s}: {mark} {tokens:>6,}t  "
                  f"{latency:>5.0f}ms  top: {ids[:2]}".replace(",", "."))
        detailed.append(row)

    print("\n" + "=" * 88)
    print("Aggregat über alle Fragen")
    print("=" * 88)
    n = len(QUESTIONS)
    print(f"{'Methode':<15} {'Recall@5':>10} {'Ø Tokens':>12} {'Ø Latenz':>10} {'Ø Kosten':>14}")
    print("-" * 88)
    summary = {}
    for name, _ in METHODS:
        m = per_method[name]
        recall = m["hits"] / n * 100
        avg_tok = sum(m["tokens"]) / len(m["tokens"]) if m["tokens"] else 0
        avg_lat = sum(m["latency"]) / len(m["latency"]) if m["latency"] else 0
        avg_cost = cost_input_usd(avg_tok)
        avg_cost_eur = avg_cost * USD_TO_EUR
        print(f"{name:<15} {recall:>9.1f}% {avg_tok:>12,.0f} {avg_lat:>9.0f}ms "
              f"{avg_cost_eur*100:>10.3f} ct".replace(",", "."))
        summary[name] = {
            "recall_at_5_pct": round(recall, 1),
            "avg_context_tokens": round(avg_tok),
            "avg_latency_ms": round(avg_lat),
            "avg_input_cost_usd": round(avg_cost, 5),
            "avg_input_cost_eur": round(avg_cost_eur, 5),
        }

    # Vergleich knowmind vs. beste Alternative
    knowmind_data = summary.get("knowmind")
    if knowmind_data:
        print("\nHebel-Vergleich (knowmind vs. jede andere Methode):")
        for name, _ in METHODS:
            if name == "knowmind":
                continue
            other = summary[name]
            d_recall = knowmind_data["recall_at_5_pct"] - other["recall_at_5_pct"]
            d_tokens = knowmind_data["avg_context_tokens"] - other["avg_context_tokens"]
            d_cost_eur = (knowmind_data["avg_input_cost_eur"] - other["avg_input_cost_eur"]) * 100
            print(f"  vs {name:>13s}: ΔRecall {d_recall:+.1f}pp · ΔTokens {d_tokens:+,d} · "
                  f"ΔKosten {d_cost_eur:+.3f} ct/Anfrage".replace(",", "."))

    out = Path(__file__).parent / "token_benchmark_result.json"
    payload = {
        "model": "claude-opus-4-7",
        "price_input_usd_per_mtok": PRICE_OPUS_INPUT_PER_MTOK,
        "k": K,
        "questions_total": n,
        "per_method": summary,
        "detailed": detailed,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRoh-Daten: {out}")


if __name__ == "__main__":
    main()
