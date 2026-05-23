# knowmind-benchmark

Öffentlich nachvollziehbarer Vergleich von [knowmind](https://knowmind.de) gegen drei realistische Retrieval-Alternativen, die ein Nutzer ohne dedizierten Memory-Layer typischerweise einsetzt:

- **ripgrep** — Volltextsuche, Top-K nach Match-Count
- **fname** — Dateinamen-Suche, Top-K nach Token-Match im Slug
- **vector_only** — pures Embedding-Retrieval, klassisches RAG ohne Hybrid
- **knowmind** — Hybrid aus BM25, Vector, Graph, Identity- und Relationship-Routing

Gemessen wird Recall@5 (lag die richtige Quelle in den Top-5?), durchschnittliche Kontext-Token-Anzahl pro Anfrage und Latenz.

## Ergebnis vom 23. Mai 2026 (Held-Out, 20 Fragen)

| Methode | Recall@5 | Ø Tokens | Ø Latenz | Ø Kosten Opus 4.7 |
|---|---|---|---|---|
| ripgrep | 55 % | 3.244 | 27 ms | 4,48 ct |
| fname | 55 % | 2.660 | 2 ms | 3,67 ct |
| vector_only | 65 % | 1.620 | 491 ms | 2,24 ct |
| **knowmind** | **80 %** | **1.790** | 475 ms | **2,47 ct** |

**knowmind: +15 pp gegen Vector-only, +25 pp gegen Volltext-/Datei­namen-Suche, bei 33–45 % weniger Kontext-Tokens als die simplen Verfahren.**

Roh-Daten: [`results/2026-05-23/`](results/2026-05-23/).

## Was die Zahlen nicht sagen

Der Test misst die hier veröffentlichte Pipeline-Konfiguration gegen den hier veröffentlichten Korpus mit den hier veröffentlichten Fragen. Bei n=20 ist das 95 %-Konfidenzintervall für einen Recall-Wert von 80 % rechnerisch etwa 56–94 % — der wahre Wert auf einem anderen Korpus oder mit anderen Fragen kann darüber oder darunter liegen.

Konkret bedeutet das:

- **Auf Ihrem eigenen Korpus** kann knowmind besser oder schlechter abschneiden. Die relevanten Variablen sind Korpus-Größe, Domänen-Mix, Frage-Charakteristik und Embedding-Qualität.
- **Andere Embedding-Modelle** ändern alle Vector-basierten Verfahren. knowmind nutzt aktuell `multilingual-e5-large` (1024-dim).
- **Andere Sprachen**: Der Test ist auf Deutsch. Englische oder andere Sprachen können andere Ergebnisse liefern.

Wenn Sie eine Gegenmessung auf Ihrem Korpus fahren möchten: das Skript ist reproduzierbar (siehe unten), Goldstandard-Anpassung dauert pro Frage ein paar Minuten.

## Reproduzieren

### Voraussetzungen

```
python >= 3.11
pip install requests tiktoken
```

Plus optional `ripgrep` (für die Methode A). Ohne ripgrep nutzt das Skript einen Python-Fallback.

### Setup

1. Holen Sie sich einen knowmind-Token: [knowmind.de/signin](https://knowmind.de/signin) → Dashboard → API-Tokens
2. Token exportieren: `export KNOWMIND_TOKEN="kmt_..."`
3. Ihre Tenant-UUID notieren (steht im Dashboard)

### Ausführen

```bash
# Train-Set (20 Fragen, für Pipeline-Iteration)
python token_benchmark.py

# Held-Out-Set (20 Fragen, zur Bestätigung — Tuning-bias-frei)
python token_benchmark_holdout.py

# Val-Set (30 Fragen, zur Hypothesen-Validierung)
python benchmark_val.py
```

Jedes Skript schreibt eine JSON-Datei mit den Roh-Ergebnissen plus einen Aggregat-Block auf STDOUT.

## Methodik

### Goldstandard

Pro Frage sind 1–4 erwartete Memory-Slugs hinterlegt. Eine Methode erzielt einen Treffer, wenn mindestens einer dieser Slugs in den IDs oder Inhalten der Top-5-Antworten enthalten ist. Slug-Match ist `_`/`-`-normalisiert.

### Token-Zählung

Mit `tiktoken` (Modell `cl100k_base` — kompatible Tokenization-Klasse für Claude- und GPT-Modelle). Die summierten Tokens der Top-5-Excerpts plus System-Prompt entsprechen dem Input, den ein Nutzer einem LLM mitschicken müsste.

### Vergleichsmethoden

| Methode | Implementation |
|---|---|
| `ripgrep` | Volltextsuche über alle `.md`-Dateien, Match-Count pro File, Top-K sortiert |
| `fname` | Suche nach Query-Tokens (≥ 3 Zeichen) im Dateinamen, Top-K nach Treffer-Anzahl |
| `vector_only` | knowmind-Recall-API mit Stage-Filter (nur Hits mit `source=vector`) |
| `knowmind` | Volle knowmind-Pipeline: BM25 + Vector + Graph + Identity-/Relationship-Stages |

### Was am Test bewusst NICHT gemessen wird

- **Antwortqualität** des LLM. Recall@5 misst, ob die richtige Quelle im Kontext landet — nicht, ob das LLM daraus eine bessere Antwort generiert. Beides korreliert, ist aber nicht dasselbe.
- **End-to-End-Token-Aufwand inkl. Antwort**. Wir messen nur den Input-Kontext, weil der den vom Memory-Layer beeinflussbaren Anteil ausmacht.
- **Cold-Cache-Latenz**. Die ersten Anfragen nach einem Container-Start sind langsamer.

## Versionierung

Jeder Benchmark-Lauf bekommt ein Datum und einen Pipeline-Commit-Hash (siehe `results/<datum>/manifest.json`). Wenn sich Pipeline, Korpus oder Modell ändern, gibt es eine neue Datei. Die Werte hier sind also reproduzierbar zu **genau diesem Stand**, nicht über die Zeit hinweg.

## Lizenz

[MIT](LICENSE) — Skript und Goldstandard dürfen kopiert, geforkt, kommerziell genutzt werden. Wir freuen uns über Pull Requests, die Gegen-Setups (eigene Korpora, andere Sprachen) hinzufügen.

## Kontakt

- **Plattform:** [knowmind.de](https://knowmind.de)
- **CLI:** `npm install -g knowmind`
- **Issues / Anregungen:** [Issues in diesem Repo](https://github.com/Schubeler-Consulting/knowmind-benchmark/issues)
- **Geschäftlich:** info@schuebeler-consulting.de
