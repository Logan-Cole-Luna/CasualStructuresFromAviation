"""
Training and evaluation script for all three models.

Models evaluated:
  1. Traditional NLP (rule-based + spaCy dependency parsing)
  2. DistilBERT transformer classifier (full fine-tuning)
  3. Knowledge graph (structural metrics)
"""
import warnings; warnings.filterwarnings('ignore')
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_data, preprocess_data
from src.traditional_nlp import batch_extract, load_nlp, CAUSAL_FORWARD, CAUSAL_BACKWARD
from src.knowledge_graph import build_graph, graph_stats, to_neo4j_cypher, visualize_subgraph
from src.transformer_classifier import NTSBClassifier, LABEL_COLS

DATA_PATH = "data/clean/cleaned_narritives_and_findings.csv"
OUTPUT_DIR = Path("outputs")
EVAL_DIR   = OUTPUT_DIR / "evaluation"
PLOTS_DIR  = OUTPUT_DIR / "plots"


def section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1. Traditional NLP evaluation
# ---------------------------------------------------------------------------

def evaluate_traditional_nlp(df: pd.DataFrame) -> dict:
    section("MODEL 1: Traditional NLP — Causal Extraction")

    # --- Rule-based on full dataset ---
    print("\n[1a] Rule-based extraction (full dataset)...")
    rule_triples = batch_extract(df, nlp=None, sample_n=None)

    # Coverage: fraction of narratives with ≥1 triple
    ev_with_triple = {t['ev_id'] for t in rule_triples}
    coverage = len(ev_with_triple) / len(df)

    # Pattern breakdown
    pattern_counts = Counter(t['relation'] for t in rule_triples)
    direction_counts = Counter(t['direction'] for t in rule_triples)

    # Triples per narrative (for narratives that have ≥1)
    per_ev = Counter(t['ev_id'] for t in rule_triples)
    densities = list(per_ev.values())

    print(f"  Total narratives:          {len(df)}")
    print(f"  Narratives with ≥1 triple: {len(ev_with_triple)} ({coverage:.1%})")
    print(f"  Total triples extracted:   {len(rule_triples)}")
    print(f"  Avg triples per narrative: {np.mean(densities):.2f}")
    print(f"  Median triples/narrative:  {np.median(densities):.1f}")
    print(f"  Direction breakdown:       {dict(direction_counts)}")
    print(f"\n  Pattern hit counts:")
    for pat, cnt in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        print(f"    '{pat}': {cnt}")

    # --- spaCy dep-parse on 500 sample ---
    print("\n[1b] spaCy dependency parsing (500-narrative sample)...")
    nlp = load_nlp()
    dep_triples = []
    dep_coverage = 0.0
    if nlp is not None:
        dep_triples = batch_extract(df, nlp=nlp, sample_n=500)
        dep_ev = {t['ev_id'] for t in dep_triples}
        dep_coverage = len(dep_ev) / min(500, len(df))

        dep_pattern_counts = Counter(t['relation'] for t in dep_triples)
        dep_per_ev = Counter(t['ev_id'] for t in dep_triples)
        dep_densities = list(dep_per_ev.values())

        print(f"  Sample size:               500")
        print(f"  Narratives with ≥1 triple: {len(dep_ev)} ({dep_coverage:.1%})")
        print(f"  Total triples extracted:   {len(dep_triples)}")
        if dep_densities:
            print(f"  Avg triples per narrative: {np.mean(dep_densities):.2f}")
        print(f"\n  Verb lemma distribution:")
        for pat, cnt in sorted(dep_pattern_counts.items(), key=lambda x: -x[1]):
            print(f"    '{pat}': {cnt}")
    else:
        print("  spaCy not available.")

    # --- Plot: pattern distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Rule-based pattern hits
    pats = list(pattern_counts.keys())
    vals = [pattern_counts[p] for p in pats]
    axes[0].barh(pats, vals, color='#2196F3')
    axes[0].set_xlabel("Triple count")
    axes[0].set_title("Rule-based: Pattern Hit Counts")
    axes[0].invert_yaxis()

    # Triple density histogram
    axes[1].hist(densities, bins=20, color='#4CAF50', edgecolor='white')
    axes[1].axvline(np.mean(densities), color='red', linestyle='--', label=f'Mean={np.mean(densities):.1f}')
    axes[1].set_xlabel("Triples per narrative")
    axes[1].set_ylabel("Narrative count")
    axes[1].set_title("Rule-based: Triple Density Distribution")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eval_traditional_nlp.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Plot saved to {PLOTS_DIR / 'eval_traditional_nlp.png'}")

    return {
        'rule_based': {
            'total_triples': len(rule_triples),
            'coverage': round(coverage, 4),
            'avg_density': round(float(np.mean(densities)), 2),
            'pattern_counts': dict(pattern_counts),
            'direction_counts': dict(direction_counts),
        },
        'dep_parsing': {
            'sample_size': 500,
            'total_triples': len(dep_triples),
            'coverage': round(dep_coverage, 4),
        },
        'all_rule_triples': rule_triples,
        'all_dep_triples': dep_triples,
    }


# ---------------------------------------------------------------------------
# 2. Transformer classifier training & evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate_transformer(df: pd.DataFrame) -> dict:
    section("MODEL 2: DistilBERT Transformer Classifier")

    try:
        import torch
    except ImportError:
        print("  torch not installed — skipping.")
        return {}

    clf = NTSBClassifier(num_labels=len(LABEL_COLS))

    print(f"\n  Training on full dataset (all {len(df)} records, 4 categories)...")
    train_ds, val_ds, test_ds, label_map = clf.prepare_data(
        df,
        text_col='narr_clean',
        label_col='top_category',
        max_samples=None,
    )

    print(f"\n  Label map: {label_map}")
    print(f"  Training for 3 epochs (batch=16, lr=2e-5)...")

    clf.train(
        train_ds, val_ds,
        epochs=3,
        batch_size=16,
        lr=2e-5,
        save_path=str(OUTPUT_DIR / "model"),
    )

    # Evaluate on held-out test set
    print("\n  Evaluating on test set...")
    results = clf.evaluate(test_ds)

    print(f"\n  Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.1f}%)")

    if 'classification_report' in results:
        report = results['classification_report']
        print("\n  Per-class metrics:")
        print(f"  {'Class':<25} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
        print("  " + "-" * 65)
        for label in LABEL_COLS:
            if label in report:
                m = report[label]
                print(f"  {label:<25} {m['precision']:>10.3f} {m['recall']:>8.3f} {m['f1-score']:>8.3f} {int(m['support']):>9}")
        if 'macro avg' in report:
            m = report['macro avg']
            print(f"  {'macro avg':<25} {m['precision']:>10.3f} {m['recall']:>8.3f} {m['f1-score']:>8.3f}")
        if 'weighted avg' in report:
            m = report['weighted avg']
            print(f"  {'weighted avg':<25} {m['precision']:>10.3f} {m['recall']:>8.3f} {m['f1-score']:>8.3f}")

    # --- Confusion matrix ---
    _plot_confusion_matrix(clf, test_ds, label_map)

    # --- Sample predictions ---
    print("\n  Sample predictions on held-out narratives:")
    sample_texts = df[df['top_category'].isin(LABEL_COLS)]['narr_clean'].sample(5, random_state=99).tolist()
    for i, text in enumerate(sample_texts):
        label, conf = clf.predict(text)
        preview = text[:80].replace('\n', ' ')
        print(f"  [{i+1}] Predicted: {label} ({conf:.1%}) | '{preview}…'")

    return results


def _plot_confusion_matrix(clf, test_ds, label_map: dict):
    """Build and save a confusion-matrix heatmap."""
    try:
        import torch
        from torch.utils.data import DataLoader
        from sklearn.metrics import confusion_matrix
    except ImportError:
        return

    loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    clf.model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(clf.device) for k, v in batch.items()}
            outputs = clf.model(**batch)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            labels = batch['labels'].cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    inv_map = {v: k for k, v in label_map.items()}
    class_names = [inv_map[i] for i in sorted(inv_map)]
    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("DistilBERT — Confusion Matrix (Test Set)")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eval_transformer_confusion.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Confusion matrix saved to {PLOTS_DIR / 'eval_transformer_confusion.png'}")


# ---------------------------------------------------------------------------
# 3. Knowledge graph evaluation
# ---------------------------------------------------------------------------

def evaluate_knowledge_graph(rule_triples: list, dep_triples: list) -> dict:
    section("MODEL 3: Knowledge Graph — Structural Evaluation")

    all_triples = rule_triples + dep_triples

    # Build graphs from each source separately and combined
    G_rules = build_graph(rule_triples)
    G_deps  = build_graph(dep_triples)
    G_all   = build_graph(all_triples)

    stats_rules = graph_stats(G_rules)
    stats_deps  = graph_stats(G_deps)
    stats_all   = graph_stats(G_all)

    def _fmt(stats, name):
        print(f"\n  [{name}]")
        print(f"    Nodes:                    {stats['num_nodes']}")
        print(f"    Edges:                    {stats['num_edges']}")
        print(f"    Density:                  {stats['density']:.6f}")
        print(f"    Weakly connected comps:   {stats['weakly_connected_components']}")
        print(f"    Top-5 causes (out-degree):")
        for node, deg in stats['top_causes'][:5]:
            short = node[:60] + ('…' if len(node) > 60 else '')
            print(f"      ({deg}) {short}")
        print(f"    Top-5 effects (in-degree):")
        for node, deg in stats['top_effects'][:5]:
            short = node[:60] + ('…' if len(node) > 60 else '')
            print(f"      ({deg}) {short}")

    _fmt(stats_rules, "Rule-based graph")
    _fmt(stats_deps,  "Dep-parse graph")
    _fmt(stats_all,   "Combined graph")

    # Node overlap between rule-based and dep-parse
    if G_rules.number_of_nodes() > 0 and G_deps.number_of_nodes() > 0:
        rule_nodes = set(G_rules.nodes())
        dep_nodes  = set(G_deps.nodes())
        overlap = len(rule_nodes & dep_nodes)
        print(f"\n  Node overlap (rule ∩ dep): {overlap} / {len(rule_nodes | dep_nodes)} unique nodes "
              f"({overlap / max(1, len(rule_nodes | dep_nodes)):.1%})")

    # Save final combined graph + Cypher
    to_neo4j_cypher(all_triples, path=str(OUTPUT_DIR / "extractions" / "neo4j_import_full.cypher"))
    visualize_subgraph(G_all, top_n=40, save_path=str(PLOTS_DIR / "eval_knowledge_graph_full.png"))
    print(f"  KG visualization saved to {PLOTS_DIR / 'eval_knowledge_graph_full.png'}")

    return {
        'rule_graph': {k: v for k, v in stats_rules.items() if k not in ('top_causes', 'top_effects', 'top_nodes_by_betweenness')},
        'dep_graph':  {k: v for k, v in stats_deps.items()  if k not in ('top_causes', 'top_effects', 'top_nodes_by_betweenness')},
        'combined_graph': {k: v for k, v in stats_all.items() if k not in ('top_causes', 'top_effects', 'top_nodes_by_betweenness')},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "extractions").mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  NTSB Causal Chain Extraction — Training & Evaluation")
    print("  Team: Madeline Gorman, Katherine Hoffsetz, Logan Luna, Stephanie Ramsey")
    print("=" * 70)

    # Load data
    print("\nLoading and preprocessing data...")
    df = load_data(DATA_PATH)
    df = preprocess_data(df)
    print(f"  Records: {len(df)}  |  Categories: {df['top_category'].value_counts().to_dict()}")

    # Model 1: Traditional NLP
    trad_results = evaluate_traditional_nlp(df)

    # Model 2: Transformer
    transformer_results = train_and_evaluate_transformer(df)

    # Model 3: Knowledge graph (uses triples from Model 1)
    kg_results = evaluate_knowledge_graph(
        trad_results['all_rule_triples'],
        trad_results['all_dep_triples'],
    )

    # Save combined evaluation report
    report = {
        'traditional_nlp': {
            'rule_based': trad_results['rule_based'],
            'dep_parsing': trad_results['dep_parsing'],
        },
        'transformer': {k: v for k, v in transformer_results.items() if k != 'classification_report'},
        'knowledge_graph': kg_results,
    }
    report_path = EVAL_DIR / "evaluation_report.json"

    def _default(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return str(obj)

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=_default)

    section("Evaluation Complete — Output Files")
    print(f"  Evaluation report:        {report_path}")
    print(f"  Traditional NLP plot:     {PLOTS_DIR / 'eval_traditional_nlp.png'}")
    print(f"  Transformer confusion:    {PLOTS_DIR / 'eval_transformer_confusion.png'}")
    print(f"  KG visualization:         {PLOTS_DIR / 'eval_knowledge_graph_full.png'}")
    print(f"  Neo4j Cypher (full):      {OUTPUT_DIR / 'extractions' / 'neo4j_import_full.cypher'}")
    print(f"  Trained model:            {OUTPUT_DIR / 'model'}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
