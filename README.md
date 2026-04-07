# Causal Structures from Aviation Accident Reports

**Team:** Madeline Gorman, Katherine Hoffsetz, Logan Luna, Stephanie Ramsey

Automatically extract and model causal chains from NTSB aviation accident narratives
using three complementary approaches: traditional NLP, a transformer classifier, and a
prompt-based LLM extractor — all feeding into a Neo4j knowledge graph.

---

## Results at a Glance

Full dataset run — all 6,059 NTSB narratives.

### Extraction Models

| Metric | Rule-based NLP | spaCy Dep-parse | Mistral-7B LLM |
|---|---|---|---|
| Narrative coverage | 44.9% | 45.5% | **97.5%** |
| **Cause-confirmed coverage** | 43.3% | 43.9% | **99.2%** |
| Total triples | 4,867 | 5,880 | **22,189** |
| Avg triples / narrative | 1.79 | 2.13 | **3.76** |
| Category alignment (vs NTSB) | 50.0% | 50.0% | 48.6% |
| Finding keyword recall | 14.5% | 14.6% | **16.7%** |
| Parse errors | 0% | 0% | 0.86% |

### DistilBERT Classifier (ground truth = NTSB finding category)

| Class | Precision | Recall | F1 | Test support |
|---|---|---|---|---|
| Personnel issues | 0.639 | **0.726** | 0.680 | 431 |
| Aircraft | **0.718** | 0.612 | 0.661 | 420 |
| Environmental issues | 0.541 | 0.569 | 0.555 | 58 |
| Not determined | 0.000 | 0.000 | 0.000 | 0 |
| **Weighted avg** | **0.669** | **0.663** | **0.663** | 909 |
| **Overall accuracy** | | | **66.3%** | |

### Knowledge Graph

| Source | Nodes | Edges | WCC | Density |
|---|---|---|---|---|
| Rule-based | 8,346 | 4,737 | 3,611 | 6.8×10⁻⁵ |
| Dep-parse | 9,835 | 5,694 | 4,147 | 5.9×10⁻⁵ |
| LLM | 30,509 | 21,358 | 12,074 | 2.3×10⁻⁵ |
| **Combined** | **39,398** | **27,038** | **13,176** | **1.7×10⁻⁵** |

---

## Project Overview

Aviation accident reports contain rich causal narratives in unstructured text:
*"fuel exhaustion resulted in a total loss of engine power, which led to an aerodynamic
stall."* Large-scale analysis of these chains is impractical by hand. This project
builds an automated pipeline that extracts cause-effect triples, classifies accident
categories, and organizes the results into a queryable knowledge graph.

The central research question is: **how does extraction quality, coverage, and causal
richness compare between rule-based NLP, a fine-tuned transformer encoder, and a
generative LLM — and how well does each align with the NTSB's official causal findings?**

---

## Dataset

- **Source:** NTSB Narrative Reports
- **Records:** 6,059 cleaned accident narratives + official `finding_description` labels
- **Ground truth:** Each accident has one or more NTSB `finding_description` entries,
  structured as a hierarchy (e.g., `Personnel issues - Task performance - Use of checklist - Pilot - C`).
  The suffix `C` marks officially confirmed causes; `F` marks findings not designated as causes.

| Category | Count |
|---|---|
| Personnel issues | 2,871 (47.4%) |
| Aircraft | 2,800 (46.2%) |
| Environmental issues | 383 (6.3%) |
| Organizational issues | 5 (0.1%) |

---

## Pipeline

```
NTSB CSV  (narr_accf + finding_description)
  │
  ├── Model 1: Rule-based NLP  ─────────────────────────┐
  │     └── spaCy Dependency Parsing                     │
  │                                                      ├──► Knowledge Graph (Neo4j)
  ├── Model 2: DistilBERT Classifier                     │
  │     └── Narrative category prediction                │
  │                                                      │
  └── Model 4: Mistral-7B LLM Extractor  ───────────────┘
        └── Prompt-based cause-effect triple extraction

  finding_description  ──► Finding Evaluator
        └── Ground-truth alignment for all extraction models
```

---

## Evaluation Metrics Explained

### Narrative Coverage

$$\text{Coverage} = \frac{\text{\# narratives with ≥1 triple}}{\text{Total narratives}} \times 100\%$$

Measures what fraction of narratives the model successfully extracted at least one
causal triple from. A narrative is "uncovered" only when the model returns 0 triples —
a silent failure.

### Cause-Confirmed Coverage (ground truth)

$$\text{CC Coverage} = \frac{\text{\# C-finding accidents with ≥1 triple extracted}}{\text{\# accidents with ≥1 NTSB C finding}} \times 100\%$$

The NTSB labels each confirmed cause with `C`. This stricter metric uses only accidents
where a cause is officially confirmed as the denominator (5,321 of 5,959 accidents),
asking: *"for accidents we know have a cause, did the model find it?"*

### Category Alignment (ground truth)

For each accident where triples were extracted, the concatenated cause+effect text is
classified into one of the four NTSB top-level categories using keyword heuristics. The
result is compared to the official finding's top-level category. Score = % correct.

### Finding Keyword Recall (ground truth)

The NTSB finding hierarchy is tokenized into concept terms (e.g., `task performance`,
`fuel management`, `spatial disorientation`). Score = average % of those tokens that
appear anywhere in the extracted cause/effect text across all evaluated accidents.

---

## Model Results

### Model 1 — Traditional NLP (Rule-based + spaCy)

| Metric | Rule-based | spaCy dep-parse |
|---|---|---|
| Narrative coverage | 44.9% | 45.5% |
| Cause-confirmed coverage | 43.3% | 43.9% |
| Total triples | 4,867 | 5,880 |
| Avg triples / narrative | 1.79 | 2.13 |
| Category alignment | 50.0% | 50.0% |
| Finding keyword recall | 14.5% | 14.6% |
| Speed | ~5,000 narratives/sec | ~35 narratives/sec |

**Key observations:**

- Coverage plateaus at ~45% regardless of dataset size — confirmed by identical numbers
  on the 5,000-sample and 6,059-sample runs. This is a structural ceiling, not a data
  size effect. The 12 trigger patterns cover only explicitly-marked causal language.
- "Resulted in" (1,889) and "due to" (1,340) account for 67% of all rule-based triples —
  the distribution is heavily long-tailed.
- Dep-parse extracts 21% more triples per covered narrative (2.13 vs 1.79) because
  grammatical relations surface additional verb-argument pairs. Node overlap between the
  two graphs is 84.9%, confirming they capture the same vocabulary through different paths.
- 43.3% cause-confirmed coverage means rule-based extraction missed the officially
  confirmed cause in more than half of NTSB-confirmed-causal accidents.

---

### Model 2 — DistilBERT Transformer Classifier

| Metric | Value |
|---|---|
| Overall test accuracy | **66.3%** |
| Weighted avg F1 | 0.663 |
| Best val accuracy | 67.8% (epoch 7 of 10) |
| Early stopping triggered | Epoch 10 (patience = 3 from epoch 7) |
| Majority-class baseline | ~47% |
| Training data | 4,237 narratives |
| Test data | 909 narratives |
| Class weighting | Aircraft 0.72×, Environmental 5.27×, Personnel 0.70× |

**Per-class performance:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Personnel issues | 0.639 | **0.726** | 0.680 | 431 |
| Aircraft | **0.718** | 0.612 | 0.661 | 420 |
| Environmental issues | 0.541 | 0.569 | 0.555 | 58 |
| Not determined | 0.000 | 0.000 | 0.000 | 0 |

**Key observations:**

- 66.3% against a 47% majority-class baseline = +19 points above chance; the model is
  learning real signal, but label ambiguity between Personnel and Aircraft is the ceiling.
- **Aircraft (high precision, lower recall):** The model is conservative — it predicts
  Aircraft confidently when mechanical vocabulary dominates, but pulls back on ambiguous
  narratives (e.g., "pilot failed to detect the mechanical failure") and misclassifies
  them as Personnel.
- **Personnel (lower precision, high recall):** The model over-predicts Personnel,
  absorbing the ambiguous Aircraft cases. Human-action words ("pilot", "crew", "decision")
  appear in both categories, biasing the model toward the more common one.
- **Environmental (F1=0.555):** Only 58 test samples despite 5.27× class weighting.
  Weather narratives contain pilot-response language that bleeds into the Personnel signal.
  The weighting is necessary but insufficient given the class imbalance.
- **Training curves:** Loss dropped monotonically (1.14 → 0.163 over 10 epochs) while
  validation accuracy oscillated between 60–68%, confirming the model hit its task
  complexity ceiling rather than continuing to improve.
- **Ground truth link:** DistilBERT's labels come directly from `finding_description`'s
  top-level category, making its test accuracy a direct measure of alignment with NTSB
  causal categorization.

---

### Model 4 — Mistral-7B-Instruct LLM Extractor

| Metric | Value |
|---|---|
| Dataset size | 6,059 narratives |
| Narrative coverage | **97.5%** (5,908 / 6,059) |
| **Cause-confirmed coverage** | **99.2%** (5,277 / 5,321) |
| Total triples | **22,189** |
| Avg triples / narrative | **3.76** |
| Parse errors (after retry) | 52 (0.86%) |
| VRAM usage (4-bit NF4) | 4.2 / 17.1 GB |
| Inference time | ~4.6 hours for 3,284 uncached narratives |
| LLM response cache | 5,959 entries |

**Top relation phrases:**

| Relation | Count | Type |
|---|---|---|
| resulted in | 7,952 | Positive causal |
| led to | 6,198 | Positive causal |
| caused | 2,044 | Positive causal |
| contributed to | 933 | Positive causal |
| precluded | 257 | Barrier failure |
| prevented | 211 | Barrier failure |
| did not prevent | 181 | Barrier failure |

**Key observations:**

- **99.2% cause-confirmed coverage** means the LLM found at least one cause-effect triple
  in 5,277 of the 5,321 accidents NTSB officially confirmed had causes. The 44 misses are
  almost entirely the 52 parse failures (invalid JSON even after retry).
- The LLM extracts **barrier failures** ("precluded", "prevented", "did not prevent") —
  a structurally distinct causal category that rule-based methods cannot express. These
  capture cases where a safety mechanism *failed to activate*, not where something actively
  caused the outcome.
- **Category alignment (48.6%)** is slightly *lower* than rule-based (50%), not because
  the LLM extracts worse causes, but because it uses richer and more varied technical
  language ("manifold pressure drop", "cyclic trim misalignment") that the keyword
  heuristic can't classify as cleanly as rule-based extractions do.
- **Environmental alignment (26.5% LLM vs 17% rule):** The LLM's strongest improvement
  over rule-based is on weather/environment causes. It correctly identifies "density
  altitude reduced climb performance" as an Environmental cause even without explicit
  causal phrases; rule-based misses this entirely.
- **Response caching** made the full-dataset run practical: 2,775 of 6,059 narratives
  were already cached from earlier runs, reducing GPU time by ~46%.

---

### Model 3 — Knowledge Graph

| Metric | Rule-based | Dep-parse | LLM | Combined |
|---|---|---|---|---|
| Nodes | 8,346 | 9,835 | 30,509 | 39,398 |
| Edges | 4,737 | 5,694 | 21,358 | 27,038 |
| Weakly connected components | 3,611 | 4,147 | 12,074 | 13,176 |
| Density | 6.8×10⁻⁵ | 5.9×10⁻⁵ | 2.3×10⁻⁵ | 1.7×10⁻⁵ |

**Top causes (combined):** `wind gust` (107), `loss of engine power` (82), `hard landing` (74),
`fuel exhaustion` (39), `spatial disorientation` (22)

**Top effects (combined):** `loss of engine power` (275), `total loss of engine power` (169),
`airplane nosed over` (160), `aerodynamic stall` (38)

**Key observations:**

- The LLM contributes 30,509 nodes — 3.6× more than rule-based alone — reflecting its
  richer and more varied extraction vocabulary. This also means more fragmentation:
  "fuel exhaustion", "fuel depletion", and "fuel starvation" remain as separate nodes
  without entity linking.
- 13,176 weakly connected components on 39,398 nodes means average component size is 3
  nodes. Most accidents form isolated causal chains with no cross-accident connectivity —
  entity disambiguation is needed before the graph becomes traversable at scale.
- `loss of engine power` appears as both a top cause (82) and top effect (275), making it
  the central hub concept in aviation accident causation — it is simultaneously the outcome
  of mechanical failures and the trigger for aerodynamic stalls and crashes.
- `accident` still appears as a top cause node, a residual noise artifact from phrases like
  *"the cause of the accident..."* — the current noise filter blocks "the accident" but not
  bare "accident".
- Node overlap between rule-based and dep-parse graphs is 84.9%, confirming they extract
  the same vocabulary through different mechanisms and combining them adds limited
  incremental value.

---

## Ground Truth Alignment Summary

All three extraction models were evaluated against the NTSB `finding_description` as
ground truth. DistilBERT's evaluation is direct (its labels are the top-level finding
categories); the extraction models are evaluated via the three alignment metrics.

| Model | Ground truth metric | Score |
|---|---|---|
| DistilBERT | Test accuracy (vs finding category) | **66.3%** |
| Rule-based | Cause-confirmed coverage | 43.3% |
| Dep-parse | Cause-confirmed coverage | 43.9% |
| **LLM** | **Cause-confirmed coverage** | **99.2%** |
| Rule-based | Category alignment | 50.0% |
| Dep-parse | Category alignment | 50.0% |
| LLM | Category alignment | 48.6% |
| Rule-based | Finding keyword recall | 14.5% |
| Dep-parse | Finding keyword recall | 14.6% |
| LLM | Finding keyword recall | **16.7%** |

**Category alignment by NTSB finding type:**

| NTSB Category | Rule-based | Dep-parse | LLM |
|---|---|---|---|
| Aircraft | 66.0% | 66.2% | 62.9% |
| Personnel issues | 36.6% | 36.3% | 37.4% |
| Environmental issues | 16.9% | 17.2% | **26.5%** |
| Organizational issues | 0% | 0% | 0% |

---

## Understanding Narrative Coverage

### Examples by Model

#### Rule-based NLP (44.9% coverage)

**Input:**
> "The pilot became spatially disoriented due to dust clouds. This led to a loss of altitude."

**Output:**
```json
[
  { "cause": "dust clouds", "relation": "due to", "effect": "spatial disorientation" },
  { "cause": "spatial disorientation", "relation": "led to", "effect": "loss of altitude" }
]
```

**Why 44.9%:** Fires only on explicit trigger phrases. Narratives with no matching phrase
(e.g., "Fuel exhaustion. Loss of engine power.") return 0 triples — 55.1% silent failures.

---

#### spaCy Dependency Parsing (45.5% coverage)

**Input:** Same narrative.

**Output:** Identical structure — grammatical relations surface the same cause-effect pairs
when explicit causal verbs are present. Marginal gain (+0.6 pp) comes from catching a
few verb-argument pairs the regex patterns miss.

---

#### Mistral-7B LLM (97.5% coverage)

**Input:** Same narrative, with extraction prompt.

**Output:**
```json
[
  { "cause": "dust clouds", "relation": "caused", "effect": "spatial disorientation" },
  { "cause": "spatial disorientation", "relation": "led to", "effect": "loss of altitude" },
  { "cause": "low altitude at onset of disorientation", "relation": "prevented", "effect": "recovery" }
]
```

**Why 97.5%:** Understands semantic intent across the full narrative. Captures implicit
causation and negative causality ("prevented recovery") invisible to pattern-based methods.
Only 0.86% parse failures remained after retry.

---

## What Comes Next

### High priority

1. **Entity linking** — "fuel exhaustion", "fuel starvation", "fuel depletion" are
   separate KG nodes. Mapping them to a shared concept (e.g., via WordNet or a
   domain ontology like CAST/HFACS) would consolidate the graph from 13,176 WCC into
   a meaningfully smaller, more traversable structure.

2. **Category-guided LLM prompting** — use DistilBERT's classification as a prefix to
   the LLM prompt ("This is a Personnel issues accident — find the specific pilot
   decision or action..."). Should improve both category alignment (~50%) and finding
   keyword recall (~16%) by focusing extraction on the right causal domain.

3. **Human evaluation** — automated metrics cannot measure extraction *precision*.
   A sample of ~50–100 triples per method, manually rated for correctness and
   specificity, is needed to complete the quality picture.

### Medium priority

4. **Expand rule patterns** — "stemmed from", "triggered by", "aggravated by",
   "following", "after" would raise the rule-based ceiling from ~45% toward ~55%
   with minimal implementation cost.

5. **KG noise filter expansion** — `"accident"` and `"pilot"` still appear as generic
   top-cause nodes. Phrase-level pattern matching (rather than exact token matching)
   would clean up these artifacts.

6. **Neo4j query examples** — the Cypher export is complete (27,171 statements).
   Writing representative queries (*"what are the most common causes of aerodynamic
   stalls?"*, *"which nodes have the highest betweenness centrality?"*) would
   demonstrate the graph's analytical value.

### Longer term

7. **Aviation domain fine-tuning** — pretraining DistilBERT on FAA Advisory Circulars
   or ASRS reports before fine-tuning on NTSB categories should push the accuracy
   ceiling above the current 66.3%.

8. **Graph Neural Networks** — once entity linking reduces fragmentation, GNNs
   (e.g., R-GCN) could learn richer accident-pathway representations for downstream
   prediction tasks.

---

## Repository Structure

```
.
├── CONFIG.conf                   # All configurable hyperparameters (sample_n=0 for full dataset)
├── main.py                       # Pipeline entry point (runs train + eval)
├── generate_plots.py             # Regenerate all plots from saved artifacts (no retraining)
├── src/
│   ├── train.py                  # Extraction + DistilBERT training
│   ├── eval.py                   # Evaluation + plot generation
│   ├── plotting.py               # All matplotlib/seaborn plot functions
│   ├── finding_evaluator.py      # Ground-truth alignment vs NTSB finding_description
│   ├── data_loader.py            # CSV loading and preprocessing
│   ├── traditional_nlp.py        # Rule-based + spaCy extraction
│   ├── transformer_classifier.py # DistilBERT fine-tuning and inference
│   ├── llm_extractor.py          # Mistral-7B prompt extraction + response cache
│   └── knowledge_graph.py        # NetworkX graph + Neo4j Cypher export
├── data/clean/                   # Cleaned NTSB CSV (not committed)
└── outputs/
    ├── model/                    # Saved DistilBERT weights + label map
    ├── evaluation/               # evaluation_report.json (all metrics)
    ├── extractions/              # llm_triples.json, llm_response_cache.json,
    │                             # neo4j_import_full.cypher, graph_stats_updated.json
    ├── training/                 # rule_triples.json, dep_triples.json, train_history.json
    └── plots/                    # All generated figures (16 PNG files)
```

---

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Full pipeline (train + eval, uses entire dataset by default)
python main.py

# Regenerate all plots from existing artifacts (no GPU needed for extraction models)
python generate_plots.py

# Run independently
python -m src.train     # extraction + DistilBERT training only
python -m src.eval      # evaluation + plots only
```

All hyperparameters are in [CONFIG.conf](CONFIG.conf). Set `sample_n = 0` to use the
entire dataset (default); set to any positive integer to subsample.
