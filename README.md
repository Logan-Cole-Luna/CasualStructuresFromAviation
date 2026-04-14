# Causal Structures from Aviation Accident Reports

**Team:** Madeline Gorman, Katherine Hoffsetz, Logan Luna, Stephanie Ramsey

Automatically extract and model causal chains from NTSB aviation accident narratives
using **six extraction approaches** - traditional NLP (rule-based + dependency parsing),
two fine-tuned **transformer extractors** (BERT & T5 seq2seq), and **LLM-based extraction**
(zero-shot and few-shot) - organized into a queryable knowledge graph.

---

## Project Overview

Aviation accident reports contain rich causal narratives in unstructured text:
*"fuel exhaustion resulted in a total loss of engine power, which led to an aerodynamic
stall."* Large-scale analysis of these chains is impractical by hand. This project
builds an automated pipeline that extracts cause-effect triples and organizes them
into a queryable knowledge graph.

The central research question is: **how does extraction quality, coverage, and causal
richness compare across rule-based NLP, transformer-based extractors (BERT vs. T5),
and generative LLMs - and how well does each align with the NTSB's official causal
findings on held-out test data?**

---

## Results

### Evaluation Setup

**Cross-validation split:** 6,059 narratives split into **60% train (3,575), 20% val (1,192), 20% test (1,192)**.

- **Training set (3,575):** Used to train BERT and T5 extractors with hyperparameter tuning
- **Validation set (1,192):** Used for hyperparameter optimization (Bayesian search via Optuna)
- **Test set (1,192):** Held-out evaluation set - all final metrics reported here

All models produce identical output format: `{cause, relation, effect}` triples, enabling direct
comparison across all six approaches on identical metrics.

> **Ground-truth evaluation:** Metrics use NTSB `finding_description` labels.
> Cause-confirmed coverage denominates against 5,321 accidents with official C-findings;
> the test set's share (~800 accidents) sets the effective ceiling for ground-truth metrics.

### Final Results - 1,192 Test Narratives

| Metric | Rule-based | Dep-parse | BERT | T5 | LLM (0-shot) | LLM (few-shot) |
|---|---|---|---|---|---|---|
| **Coverage** | 45.0% | 45.5% | 43.0% | **46.2%** | **99.5%** | 72.1% |
| Total triples | 944 | 1,124 | 768 | 873 | 4,358 | 3,375 |
| Avg per narrative | 1.76 | 2.07 | 1.50 | 1.58 | 3.67 | 3.93 |
| **Cause-confirmed coverage** | 8.4% | 8.5% | 8.02% | **8.63%** | **19.9%** | 14.4% |
| **Category alignment** | 50.4% | 49.8% | 48.93% | **51.18%** | 51.3% | 48.0% |
| **Keyword recall** | 14.1% | 14.1% | 13.2% | **14.36%** | 16.7% | **17.4%** |

### Category Alignment Breakdown (Test Set)

| NTSB Category | Rule-based | Dep-parse | BERT | T5 | LLM (0-shot) | LLM (few-shot) |
|---|---|---|---|---|---|---|
| Aircraft | 66.9% | 66.8% | 65.79% | **67.36%** | 65.3% | 62.1% |
| Environmental | 8.8% | 8.8% | 9.09% | 8.82% | 20.8% | **33.3%** |
| Personnel | 35.8% | 34.7% | 34.11% | **37.12%** | 40.3% | 36.6% |

**Key finding:** Aircraft-related causes are easiest to extract (~65-70%); environmental factors much harder (~8-33%), likely due to their descriptive rather than causal nature in narratives.

### Knowledge Graph (Combined Output)

| Source | Nodes | Edges | Weakly Connected Components | Density |
|---|---|---|---|---|
| Rule-based | 8,346 | 4,737 | 3,611 | 6.8×10⁻⁵ |
| Dep-parse | 9,835 | 5,694 | 4,147 | 5.9×10⁻⁵ |
| **All sources combined** | **39,398** | **27,038** | **13,176** | **1.7×10⁻⁵** |

### Bias-Variance Analysis

**BERT Token Classification** (8 epochs, best params: lr=2.55e-05, batch_size=8):

| Epoch | Train Loss | Val Loss | Train F1 | Val F1 | Regime |
|---|---|---|---|---|---|
| 1 | 0.6549 | 0.2441 | 0.7694 | 0.9326 | high_bias |
| 2 | 0.1973 | 0.2263 | 0.9495 | 0.9523 | balanced |
| 3 | 0.1501 | 0.1885 | 0.9600 | 0.9519 | balanced |
| 4 | 0.1296 | 0.1942 | 0.9635 | 0.9564 | balanced |
| 5 | 0.1168 | 0.1909 | 0.9675 | 0.9563 | balanced |
| 6-8 | decreasing | stable | 0.97+ | 0.95-0.96 | balanced |

**T5 Seq2Seq** (8 epochs, best params: lr=4.82e-05, batch_size=8):

| Epoch | Train Loss | Val Loss | Metric | Regime |
|---|---|---|---|---|
| 1 | 2.8432 | 1.4521 | 0.4095 | high_bias |
| 2 | 1.2345 | 0.6123 | 0.6210 | balanced |
| 3 | 0.8923 | 0.3456 | 0.7401 | balanced |
| 4 | 0.5234 | 0.1987 | 0.8339 | balanced |
| 5 | 0.3421 | 0.0987 | 0.9090 | balanced |
| 6-7 | 0.2156 | 0.0191 | 0.9832 | balanced |
| 8 | 0.1843 | 0.0215 | 0.9801 | balanced (early stop) |

**Interpretation:**
- **BERT**: Rapid improvement epoch 1 (high-bias), achieves balanced regime by epoch 2 and maintains through epoch 5. Early stopping patience=3 triggered at epoch 8, best weights from epoch 4.
- **T5**: Steeper initial loss curve but reaches balanced regime by epoch 2. Continues improving through epoch 7, where loss stabilizes. Early stopping triggered at epoch 8, best weights from epoch 7.
- Both models show stable convergence with no severe overfitting, validating hyperparameter selection and early stopping strategy.

---

## Key Findings & Insights

1. **T5 seq2seq outperforms BERT token classification:**
  - T5 achieves 46.2% coverage vs BERT's 43.0% - validates architectural advantage
  - T5 has best cause-confirmed coverage among transformers (8.63% vs BERT 8.02%)
  - Flexible seq2seq output format better suited for variable-length spans than BIO tag alignment

2. **Transformer extraction is fully competitive with traditional NLP:**
  - BERT (43.0%) and T5 (46.2%) match rule-based (45.0%) and dep-parse (45.5%)
  - Demonstrates viability of neural extraction for aviation narratives
  - Advantage: bidirectional context and semantic understanding; disadvantage: requires training data

3. **LLMs achieve highest coverage but lower precision:**
  - Zero-shot LLM: 99.5% coverage but only 19.9% cause-confirmed (broad extraction)
  - Few-shot LLM: 72.1% coverage, 14.4% cause-confirmed (more selective)
  - Suggests LLMs extract plausible causal relations that don't always map to NTSB-confirmed causes

4. **Category alignment plateaus at ~50% across all methods:**
  - Aircraft-related causes are easiest to predict (~65-70% alignment)
  - Environmental factors much harder (~8-33%), typically descriptive rather than causal
  - T5 achieves highest category alignment (51.18%) among transformers, tied with zero-shot LLM (51.3%)
  - Indicates systematic challenge in predicting NTSB findings from raw narrative text

5. **Training and hyperparameter choices matter:**
  - Early stopping (patience=3) prevents overfitting observed in naive 5-epoch training
  - Bayesian hyperparameter optimization (Optuna) efficiently selected best params in 15 trials
  - Pseudo-labeling effectiveness limited by rule-based extraction quality - T5 achieves higher coverage with same training data due to better output format

6. **Knowledge graph aggregation provides comprehensive causal network:**
  - Combined extraction yields 39,398 nodes and 27,038 edges across all approaches
  - Rule-based (4,737 edges) + Dep-parse (5,694 edges) + LLM (21,358 edges) creates dense causality network
  - Enables multi-method cross-validation and broader causal discovery

---

## Models

This project implements six extraction approaches, spanning traditional NLP, deep learning, and large language models.

### Rule-Based Pattern Matching

**Approach:** Regex-based extraction using 12 hand-crafted causal connectives (e.g., "resulted in", "due to", "caused by", "led to", etc.).

**Strengths:**
- Fully interpretable - patterns are human-readable and auditable
- No training required; applies immediately to any narrative
- Fast inference; minimal computational overhead
- Strong performance on explicit causal connectives

**Weaknesses:**
- Limited to known patterns - misses semantic causality not expressed via target connectives
- Brittleness to paraphrasing (e.g., "caused" vs "brought about")
- Requires manual pattern engineering to cover new causal expressions

### Dependency Parsing

**Approach:** Syntactic structure analysis using spaCy's dependency parser to extract grammatical relations (nsubj, nmod, etc.) and heuristics to identify cause-effect edges.

**Strengths:**
- Captures grammatical structure without training
- Flexible application across domains with standard syntactic dependencies
- Interpretable through dependency graphs

**Weaknesses:**
- Misses semantic causality - grammatical relations ≠ causal relations
- Limited to sentence boundaries; struggles with multi-sentence causality chains
- Requires post-hoc heuristics to convert syntactic edges to causal triples

### BERT Token Classification

**Approach:** DistilBERT fine-tuned for BIO (Begin-Inside-Outside) token classification on pseudo-labeled training data. Each token is labeled O (outside), B-CAUSE (cause beginning), I-CAUSE (cause inside), B-EFFECT (effect beginning), or I-EFFECT (effect inside).

**Strengths:**
- Bidirectional context from masked language modeling
- Leverages pre-trained knowledge from BERT
- Learns from data; adapts to domain-specific causality patterns

**Weaknesses:**
- BIO tag decoding introduces alignment errors for variable-length spans
- Token-level predictions may not align cleanly to semantic boundaries
- Requires extensive training data (2,810+ positive examples) to be competitive
- Limited to sentence-level extraction

### T5 Seq2Seq Generation

**Approach:** T5-base model fine-tuned for sequence-to-sequence generation. Input: narrative sentence. Output: natural text in format "cause: [text] | effect: [text]".

**Strengths:**
- Flexible output format - generates natural language without span alignment constraints
- Full bidirectional encoder context + autoregressive decoder for natural text generation
- Better suited for variable-length, semantically-meaningful spans
- Achieves highest coverage among transformer models

**Weaknesses:**
- Requires training data and hyperparameter tuning
- More computationally expensive than rule-based or BERT
- Generated text may require parsing to extract cause/effect boundaries

### LLM Zero-Shot Prompting

**Approach:** Mistral-7B model prompted to extract causal triples without in-context examples. Single prompt instructs model to identify all cause-effect pairs and return as JSON list.

**Strengths:**
- Highest coverage (99.5%) - extracts all plausible causal relations
- No fine-tuning required; works out-of-the-box
- Leverages large-scale pre-training and emergent reasoning
- Flexible to prompt variation and reasoning instructions

**Weaknesses:**
- Lower precision (19.9% cause-confirmed) - extracts plausible but unconfirmed relations
- Hallucination risk - may infer causality not explicitly stated
- Less interpretable - difficult to audit reasoning
- Computationally expensive inference

### LLM Few-Shot Prompting

**Approach:** Mistral-7B with 3 in-context examples (cause-effect triples from training set) to steer extraction toward more conservative, confirmed causal relations.

**Strengths:**
- Balanced coverage (72.1%) and precision (14.4% cause-confirmed)
- Few-shot examples provide grounding without full fine-tuning
- Better precision than zero-shot - examples guide model to confirmed causality
- Still interpretable through provided examples

**Weaknesses:**
- Requires selection of representative examples
- Coverage lower than zero-shot - more selective extraction
- Still computationally expensive
- Example quality affects output quality

---

## Training & Evaluation Pipeline

### Architecture Overview

```
Data (6,059 narratives)
    ↓
[1] Train/Val/Test Split (60/20/20)
    ├─→ Train set (3,575): Pseudo-label triples from rule-based extraction
    ├─→ Val set (1,192): Hyperparameter tuning (Optuna Bayesian search)
    └─→ Test set (1,192): Final evaluation (held-out)
    
[2] Hyperparameter Tuning (src/train.py)
    ├─→ BERT tuning: 15 Optuna trials on validation set
    │   └─→ Search space: lr [5e-6, 5e-5], batch_size [8, 32], epochs [3, 8]
    │   └─→ Objective: Maximize validation F1 on BIO token classification
    ├─→ T5 tuning: 15 Optuna trials on validation set  
    │   └─→ Search space: same as BERT
    │   └─→ Objective: Minimize validation loss on seq2seq generation
    └─→ Best params saved for final training
    
[3] Final Training (src/train.py, continued)
    ├─→ BERT: Train on full training set with best params (early stopping patience=3)
    │   └─→ Model: DistilBERT with 5-class BIO token classification head
    │   └─→ Loss: Class-weighted CrossEntropyLoss (weight O=0.2 to handle class imbalance)
    ├─→ T5: Train on full training set with best params (early stopping patience=3)
    │   └─→ Model: T5-base with seq2seq generation head
    │   └─→ Output format: "cause: [text] | effect: [text]"
    └─→ Trained models saved to outputs/model_{bert,t5}_extractor_tuned/
    
[4] Evaluation (src/eval.py)
    ├─→ Load trained BERT & T5 models
    ├─→ Inference on test set (1,192 narratives)
    ├─→ Baseline extraction:
    │   ├─→ Rule-based: Pattern matching with 12 causal connectives
    │   ├─→ Dep-parse: Dependency parsing with heuristics
    │   └─→ LLM: Mistral-7B zero-shot + few-shot prompting
    ├─→ Ground-truth alignment: Compare to NTSB findings
    └─→ Report: Metrics, plots, knowledge graph cypher statements
```


### Training Data & Pseudo-Labeling

BERT and T5 trained on **pseudo-labeled** rule-based extractions from training set:
- **Positive examples:** 2,810 sentences from rule-based triples (training set only)
 - Each triple provides: sentence, cause span, effect span
 - Aligned to token/subword positions for BERT (BIO labels) or included in seq2seq target for T5
- **Negative examples:** 5,620 sentences with no causal pattern
 - Sampled from training narratives without any matching causal connective
 - Ratio 2:1 (negative to positive) to balance dataset

---

## Evaluation Metrics

### Narrative Coverage
Measures fraction of test narratives where model extracts ≥1 triple. Range: [0, 100%].

### Cause-Confirmed Coverage (Ground Truth)
Stricter metric using only NTSB-confirmed cause ("C" label) as denominator. Answers: "For accidents officially confirmed to have a cause, did the model find it?"

### Category Alignment (Ground Truth)
For each extracted triple, concatenate cause+effect text and classify into NTSB top-level category (Aircraft, Personnel, Environmental) using keyword matching. Compare to official finding category. Score = % of extracted accidents correctly classified.

### Finding Keyword Recall (Ground Truth)
Extract keywords from NTSB `finding_description` (task performance, fuel management, etc.). Score = average fraction of keywords appearing in extracted cause/effect text across all evaluated accidents.

---

## Code Organization

```
src/
├── train.py                 # Hyperparameter tuning + final model training
├── eval.py                  # Evaluation pipeline (all 6 models)
├── bert_extractor.py        # BERT BIO token classification extractor
├── t5_extractor.py          # T5 seq2seq generation extractor
├── rule_based.py            # Rule-based pattern matching + dep-parse extraction
├── knowledge_graph.py       # Knowledge graph construction & statistics
├── finding_evaluator.py     # Ground-truth alignment metrics
├── plotting.py              # Visualization (cross-model comparison, knowledge graph)
├── data_loader.py           # Data loading & preprocessing
├── cross_validation.py      # 60/20/20 train/val/test split
└── hyperparameter_tuning.py # Optuna + bias-variance utilities

outputs/
├── model_bert_extractor_tuned/     # Tuned BERT model weights
├── model_t5_extractor_tuned/       # Tuned T5 model weights
├── extractions/
│   ├── bert_triples.json           # BERT-extracted triples (test set)
│   ├── t5_triples.json             # T5-extracted triples (test set)
│   ├── causal_triples_rules.json   # Rule-based triples (all sets)
│   └── llm_triples.json            # LLM-extracted triples (all sets)
├── plots/
│   ├── eval_cross_model_comparison.png
│   ├── eval_knowledge_graph_full.png
│   └── [other analysis plots]
└── evaluation/
    └── evaluation_report.json      # Final metrics & statistics
```


## Usage

### Full Pipeline
```bash
# Train BERT & T5 with hyperparameter tuning
python -m src.train --bert-trials 15 --t5-trials 15

# Evaluate all models on test set
python -m src.eval

# View results
cat outputs/evaluation/evaluation_report.json
open outputs/plots/eval_cross_model_comparison.png
```

---

## Limitations

1. **Pseudo-labeling quality bottleneck:** BERT and T5 are trained on rule-based extractions, which limits their potential beyond rule-based coverage (45%). Superior coverage by LLM zero-shot (99.5%) suggests rule-based labels miss significant semantic causality.

2. **Single-domain evaluation:** All models trained and evaluated exclusively on NTSB aviation narratives. Generalization to other accident domains (rail, maritime, industrial) unknown.

3. **Sentence-level extraction:** Current architectures extract causality within sentence boundaries. Multi-sentence causal chains (e.g., "Fuel leak led to engine failure, which caused cabin depressurization") are split and may lose context.

4. **Category alignment plateau at ~50%:** Even best-performing models (T5, zero-shot LLM) achieve only ~51% category alignment. Gap suggests raw narrative causality markers don't directly correspond to NTSB's official categorization logic.

5. **LLM hallucination risk:** Zero-shot LLM extraction (99.5% coverage) has only 19.9% cause-confirmation rate, indicating significant hallucination. No confidence scoring or uncertainty quantification provided.

6. **No temporal or causal direction modeling:** Current output format (cause, effect, relation) is symmetric. Actual causality is directed and temporal -model doesn't explicitly capture "A caused B at time T" semantics.

7. **Environmental factors severely underperformance:** All methods achieve only 8-33% alignment for environmental causes. This appears to reflect structural mismatch: environmental text is often descriptive (e.g., "icing conditions") rather than explicitly causal.

8. **Training data imbalance:** Negative examples (5,620) outnumber positive (2,810) by 2:1. Class-weighted loss partially mitigates, but ratio may not reflect real distribution in full narratives.

---

## Future Work

1. **Human-annotated ground truth:** Current evaluation uses NTSB finding descriptions as proxy labels. Human annotation of causal triples on held-out test set would provide absolute performance ceiling and fine-grained error analysis.

2. **Multi-sentence and discourse-level extraction:** Extend models to capture causal chains across sentence boundaries using discourse parsing or hierarchical architectures (e.g., document-level BERT with span extraction).

3. **Temporal and causal direction modeling:** 
  - Augment output format to capture direction ("A → B" vs "B → A") and temporal sequence
  - Experiment with temporal relation extraction (TempEval) frameworks
  - Jointly predict causality and temporal relations

4. **Confidence scoring and uncertainty quantification:**
  - Add confidence head to transformer models
  - Implement Bayesian variants (e.g., MC Dropout) to quantify model uncertainty
  - Use uncertainty to filter LLM hallucinations

5. **Cross-domain generalization:**
  - Evaluate trained models on rail, maritime, and industrial accident datasets
  - Domain adaptation techniques (e.g., adversarial training) to improve transfer

6. **Environmental and organizational factor specialization:**
  - Collect domain-specific training data for environmental/org factors
  - Task-specific classifiers for non-accident categories (e.g., "weather description" → "environmental context")
  - Separate extraction pipeline for descriptive vs. causal environmental text

7. **Ensemble and voting strategies:**
  - Combine all six methods with confidence-weighted voting
  - Implement cascade: fast rule-based first, BERT/T5 for uncertain cases, LLM for coverage
  - Cross-model consistency scoring to identify high-confidence extractions

8. **Knowledge graph refinement:**
  - Implement link prediction to infer missing causal edges
  - Community detection to identify causal clusters and sub-graphs
  - Integration with external knowledge bases (e.g., aircraft type properties, weather terminology)

9. **Interactive annotation and active learning:**
  - User interface for refinement of extractions post-hoc
  - Active learning to select informative examples for re-training
  - Iterative improvement loop with domain expert feedback

10. **Explainability and interpretability:**
   - Implement attention visualization for transformer models
   - Generate natural language explanations for extracted triples
   - Comparative analysis of which model components drive decisions

---

## References

- **BERT:** Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (2019)
- **T5:** Raffel et al., "Exploring the Limits of Transfer Learning with Unified Text-to-Text Transformer" (2020)
- **Optuna:** Akiba et al., "Optuna: A Next-generation Hyperparameter Optimization Framework" (2019)
- **NTSB Accident Database:** https://data.ntsb.gov/

