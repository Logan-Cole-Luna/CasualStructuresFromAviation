"""
Plotting utilities for NTSB Causal Chain Extraction.
Standardized functions for all model comparison plots.

No torch / transformers imports - purely matplotlib + seaborn + numpy.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# =========================================================================
# Helpers
# =========================================================================

def _save(fig, plots_dir: Path, filename: str):
    """Save figure and close."""
    path = plots_dir / filename
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved -> {path}")


def _model_colors():
    """Return standard color palette for 6 models."""
    return {
        'Rule-based': '#2196F3',      # Blue
        'Dep-parse': '#4CAF50',        # Green
        'BERT': '#FF9800',             # Orange
        'T5': '#FFC107',               # Amber
        'LLM (0-shot)': '#9C27B0',     # Purple
        'LLM (few-shot)': '#e91e63',   # Pink
    }


def _get_color_list(models: List[str]) -> List[str]:
    """Get colors for a list of model names."""
    palette = _model_colors()
    return [palette.get(m, '#666666') for m in models]


# =========================================================================
# Standardized Extraction Metrics
# =========================================================================

def _compute_extraction_stats(triples: List[dict]) -> Tuple[int, int, float]:
    """
    Compute coverage, total triples, and avg density for a triple list.
    Returns: (num_narratives, total_triples, avg_per_narrative)
    """
    if not triples:
        return 0, 0, 0.0
    narratives = len({t['ev_id'] for t in triples})
    total = len(triples)
    avg = total / max(1, narratives)
    return narratives, total, avg


def plot_cross_model_comparison_all_six(
    models_dict: Dict[str, List[dict]],
    sample_n: int,
    plots_dir: Path,
):
    """
    Coverage, density, and total yield bars for all 6 models.
    models_dict: {'Rule-based': [...], 'BERT': [...], ...}
    """
    model_names = list(models_dict.keys())
    stats = [_compute_extraction_stats(models_dict[name]) for name in model_names]

    narratives_list = [s[0] for s in stats]
    triples_list = [s[1] for s in stats]
    avg_density = [s[2] for s in stats]
    coverage_pct = [n / sample_n * 100 for n in narratives_list]

    colors = _get_color_list(model_names)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Coverage %
    bars = axes[0].bar(model_names, coverage_pct, color=colors, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, coverage_pct):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}%',
                     ha='center', va='bottom', fontweight='bold', fontsize=10)
    axes[0].set_ylabel('Narratives with ≥1 triple (%)')
    axes[0].set_title('Extraction Coverage')
    axes[0].set_ylim(0, max(coverage_pct) * 1.15)
    axes[0].grid(True, axis='y', alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)

    # Average density
    bars = axes[1].bar(model_names, avg_density, color=colors, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, avg_density):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.05, f'{val:.2f}',
                     ha='center', va='bottom', fontweight='bold', fontsize=10)
    axes[1].set_ylabel('Avg triples per narrative')
    axes[1].set_title('Extraction Density')
    axes[1].grid(True, axis='y', alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)

    # Total triples
    bars = axes[2].bar(model_names, triples_list, color=colors, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, triples_list):
        axes[2].text(bar.get_x() + bar.get_width()/2, val + 50, f'{val:,}',
                     ha='center', va='bottom', fontweight='bold', fontsize=9)
    axes[2].set_ylabel('Total triples')
    axes[2].set_title('Triple Yield')
    axes[2].grid(True, axis='y', alpha=0.3)
    axes[2].tick_params(axis='x', rotation=45)

    plt.suptitle('Cross-Model Comparison - All Six Extraction Methods (Test Set)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, plots_dir, 'eval_cross_model_comparison_all_six.png')


# =========================================================================
# Relation Phrases
# =========================================================================

def plot_top_relation_phrases(
    models_dict: Dict[str, List[dict]],
    plots_dir: Path,
    top_n: int = 10,
):
    """
    Six-panel horizontal bar chart: top relation phrases for all six models.
    models_dict: {'Rule-based': [...], 'BERT': [...], ...}
    """
    model_names = list(models_dict.keys())
    colors = _get_color_list(model_names)

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    axes_flat = axes.flatten()

    for ax, name, color in zip(axes_flat, model_names, colors):
        triples = models_dict[name]
        if not triples:
            ax.set_title(f'{name}\n(no triples)')
            ax.axis('off')
            continue

        counts = Counter(t['relation'] for t in triples)
        top = sorted(counts.items(), key=lambda x: -x[1])[:top_n]
        labels = [p for p, _ in top]
        vals = [v for _, v in top]
        total = sum(counts.values())

        bars = ax.barh(labels, vals, color=color, alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            pct = val / max(1, total) * 100
            ax.text(val + max(vals) * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:,} ({pct:.1f}%)', va='center', fontsize=8)

        ax.invert_yaxis()
        ax.set_xlabel('Triple count')
        n_ev = len({t['ev_id'] for t in triples})
        ax.set_title(
            f'{name}\n{len(triples):,} triples - {n_ev} narratives\n'
            f'top {top_n} of {len(counts)} unique relations',
            fontsize=10, fontweight='bold',
        )
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_xlim(right=max(vals) * 1.25 if vals else 1)

    plt.suptitle('Top Relation Phrases - Test Set (1,192 narratives) - All Six Models',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, plots_dir, 'eval_top_relation_phrases.png')


# =========================================================================
# Finding Alignment (Ground Truth)
# =========================================================================

def plot_finding_alignment(
    alignment_results: List[dict],
    plots_dir: Path,
):
    """
    3-panel figure: category alignment, cause-confirmed coverage, keyword recall.
    alignment_results: list of dicts from finding_evaluator
    """
    if not alignment_results:
        return

    labels = [r['label'] for r in alignment_results]
    colors = _get_color_list(labels)

    cat_aln = [r['category_alignment_score'] * 100 for r in alignment_results]
    cc_cov = [r['cause_confirmed_coverage'] * 100 for r in alignment_results]
    kw_rec = [r['finding_keyword_recall'] * 100 if r['finding_keyword_recall'] is not None else None
              for r in alignment_results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    def _bar_with_na(ax, vals, model_labels, model_colors, title, ylabel):
        """Bar chart that skips None values."""
        x = np.arange(len(model_labels))
        w = 0.65
        for i, (val, _, col) in enumerate(zip(vals, model_labels, model_colors)):
            if val is None:
                ax.bar(i, 0, w, color='#e0e0e0', alpha=0.5, edgecolor='white')
                ax.text(i, 3, 'N/A', ha='center', va='bottom', fontsize=9, color='#999', fontweight='bold')
            else:
                ax.bar(i, val, w, color=col, alpha=0.85, edgecolor='white')
                ax.text(i, val + 0.5, f'{val:.1f}%', ha='center', va='bottom',
                        fontweight='bold', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=10, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, 115)
        ax.grid(True, axis='y', alpha=0.3)

    _bar_with_na(axes[0], cat_aln, labels, colors,
                 'Category Alignment\n(predicted cat == NTSB finding cat)', 'Accuracy (%)')
    _bar_with_na(axes[1], cc_cov, labels, colors,
                 'Cause-Confirmed Coverage\n(C-findings only)', '% of C-finding accidents covered')
    _bar_with_na(axes[2], kw_rec, labels, colors,
                 'Finding Keyword Recall\n(% finding tokens in text)', 'Avg recall (%)')

    for ax, n_items in zip(axes, [
        [f'n={r["category_alignment_n"]}' for r in alignment_results],
        [f'{r["cause_confirmed_n"]}/{r["cause_confirmed_denom"]}' for r in alignment_results],
        [f'n={r["keyword_recall_n"]}' if r['finding_keyword_recall'] is not None else 'N/A'
         for r in alignment_results],
    ]):
        for i, label in enumerate(n_items):
            ax.text(i, ax.get_ylim()[0] + 2, label, ha='center', va='bottom', fontsize=7, color='grey')

    plt.suptitle('NTSB Finding-Alignment Evaluation - Ground Truth Comparison (Unified Test Set)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, plots_dir, 'eval_finding_alignment.png')


# =========================================================================
# Training Curves (Load from Tuning Results)
# =========================================================================

def plot_training_loss_curves(tuning_results: dict, plots_dir: Path):
    """
    Plot training loss curves for BERT and T5 from tuning results.
    tuning_results: dict with 'bert' and 't5' keys
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # BERT
    if 'bert' in tuning_results and 'training_history' in tuning_results['bert']:
        bert_hist = tuning_results['bert']['training_history']
        train_loss = bert_hist.get('train_loss', [])
        # Extract val_loss from bias_variance_logs
        val_loss = [log['val_loss'] for log in bert_hist.get('bias_variance_logs', [])]
        if train_loss and val_loss:
            epochs = list(range(1, len(train_loss) + 1))
            axes[0].plot(epochs, train_loss, 'o-', color='#FF9800', label='Train Loss', linewidth=2)
            axes[0].plot(epochs, val_loss, 's-', color='#FF6F00', label='Val Loss', linewidth=2)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('BERT - Training Loss Curve')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

    # T5
    if 't5' in tuning_results and 'training_history' in tuning_results['t5']:
        t5_hist = tuning_results['t5']['training_history']
        train_loss = t5_hist.get('train_loss', [])
        val_loss = t5_hist.get('val_loss', [])
        if train_loss and val_loss:
            epochs = list(range(1, len(train_loss) + 1))
            axes[1].plot(epochs, train_loss, 'o-', color='#FFC107', label='Train Loss', linewidth=2)
            axes[1].plot(epochs, val_loss, 's-', color='#FF6F00', label='Val Loss', linewidth=2)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('T5 - Training Loss Curve')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

    plt.suptitle('Training Loss Curves - Best Hyperparameter Trials', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, plots_dir, 'eval_training_loss_curves.png')


def plot_training_metrics(tuning_results: dict, plots_dir: Path):
    """
    Plot training accuracy (F1 for BERT, metric for T5) from tuning results.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # BERT F1
    if 'bert' in tuning_results and 'training_history' in tuning_results['bert']:
        bert_hist = tuning_results['bert']['training_history']
        train_f1 = bert_hist.get('train_f1', [])
        val_f1 = bert_hist.get('val_f1', [])
        if train_f1 and val_f1:
            epochs = list(range(1, len(train_f1) + 1))
            axes[0].plot(epochs, train_f1, 'o-', color='#FF9800', label='Train F1', linewidth=2)
            axes[0].plot(epochs, val_f1, 's-', color='#FF6F00', label='Val F1', linewidth=2)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('F1 Score')
            axes[0].set_title('BERT - Token Classification F1')
            axes[0].set_ylim([0.7, 1.0])
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

    # T5 Metric - compute from val_loss
    if 't5' in tuning_results and 'training_history' in tuning_results['t5']:
        t5_hist = tuning_results['t5']['training_history']
        val_loss = t5_hist.get('val_loss', [])
        if val_loss:
            epochs = list(range(1, len(val_loss) + 1))
            val_metric = [1 / (1 + loss) for loss in val_loss]
            axes[1].plot(epochs, val_metric, 's-', color='#FFC107', label='Val Metric (1/(1+loss))', linewidth=2)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Metric Value')
            axes[1].set_title('T5 - Generation Quality Metric')
            axes[1].set_ylim([0, 1.0])
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

    plt.suptitle('Training Metrics - Best Hyperparameter Trials', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, plots_dir, 'eval_training_metrics.png')


def plot_bias_variance_tradeoff(tuning_results: dict, plots_dir: Path):
    """
    Plot bias-variance tradeoff (train vs val loss gap) for BERT and T5.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # BERT
    if 'bert' in tuning_results and 'training_history' in tuning_results['bert']:
        bert_hist = tuning_results['bert']['training_history']
        train_loss = bert_hist.get('train_loss', [])
        bv_logs = bert_hist.get('bias_variance_logs', [])
        val_loss = [log['val_loss'] for log in bv_logs]
        if train_loss and val_loss:
            epochs = list(range(1, len(train_loss) + 1))
            gap = np.array(val_loss) - np.array(train_loss)

            colors_bert = []
            for g in gap:
                if g < -0.05:
                    colors_bert.append('#e74c3c')  # Red - underfitting
                elif g > 0.05:
                    colors_bert.append('#f39c12')  # Orange - overfitting
                else:
                    colors_bert.append('#27ae60')  # Green - balanced

            axes[0].bar(epochs, gap, color=colors_bert, alpha=0.85, edgecolor='white')
            axes[0].axhline(0, color='black', linestyle='-', linewidth=1)
            axes[0].axhline(0.05, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Overfitting threshold')
            axes[0].axhline(-0.05, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Underfitting threshold')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss Gap (Val - Train)')
            axes[0].set_title('BERT - Bias-Variance Tradeoff')
            axes[0].legend()
            axes[0].grid(True, axis='y', alpha=0.3)

    # T5
    if 't5' in tuning_results and 'training_history' in tuning_results['t5']:
        t5_hist = tuning_results['t5']['training_history']
        train_loss = t5_hist.get('train_loss', [])
        val_loss = t5_hist.get('val_loss', [])
        if train_loss and val_loss:
            epochs = list(range(1, len(train_loss) + 1))
            gap = np.array(val_loss) - np.array(train_loss)

            colors_t5 = []
            for g in gap:
                if g < -0.01:
                    colors_t5.append('#e74c3c')  # Red - underfitting
                elif g > 0.01:
                    colors_t5.append('#f39c12')  # Orange - overfitting
                else:
                    colors_t5.append('#27ae60')  # Green - balanced

            axes[1].bar(epochs, gap, color=colors_t5, alpha=0.85, edgecolor='white')
            axes[1].axhline(0, color='black', linestyle='-', linewidth=1)
            axes[1].axhline(0.01, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Overfitting threshold')
            axes[1].axhline(-0.01, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Underfitting threshold')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss Gap (Val - Train)')
            axes[1].set_title('T5 - Bias-Variance Tradeoff')
            axes[1].legend()
            axes[1].grid(True, axis='y', alpha=0.3)

    plt.suptitle('Bias-Variance Tradeoff - Training Dynamics', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, plots_dir, 'eval_bias_variance_tradeoff.png')


# =========================================================================
# Traditional NLP and LLM (single-model plots)
# =========================================================================

def plot_traditional_nlp(
    rule_triples: List[dict],
    dep_triples: List[dict],
    sample_n: int,
    plots_dir: Path,
):
    """Overview of rule-based and dep-parse extraction."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Rule pattern counts
    pattern_counts = Counter(t['relation'] for t in rule_triples)
    sorted_pats = sorted(pattern_counts.items(), key=lambda x: -x[1])
    pats = [p for p, _ in sorted_pats]
    vals = [v for _, v in sorted_pats]
    bars = axes[0, 0].barh(pats, vals, color='#2196F3', alpha=0.85)
    for bar, val in zip(bars, vals):
        axes[0, 0].text(val + 5, bar.get_y() + bar.get_height()/2, str(val), va='center', fontsize=8)
    axes[0, 0].set_xlabel('Count')
    axes[0, 0].set_title('Rule-based: Top Relation Patterns')
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(True, axis='x', alpha=0.3)

    # Density distribution
    per_ev_rule = Counter(t['ev_id'] for t in rule_triples)
    densities = list(per_ev_rule.values())
    axes[0, 1].hist(densities, bins=20, color='#4CAF50', edgecolor='white', alpha=0.85)
    axes[0, 1].axvline(np.mean(densities), color='red', linestyle='--', linewidth=1.5, label=f'Mean={np.mean(densities):.1f}')
    axes[0, 1].set_xlabel('Triples per narrative')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Rule-based: Density Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Coverage comparison
    ev_rule = len(per_ev_rule)
    ev_dep = len({t['ev_id'] for t in dep_triples})
    methods = ['Rule-based', 'Dep-parse']
    covered = [ev_rule, ev_dep]
    not_covered = [sample_n - c for c in covered]
    x = np.arange(len(methods))
    axes[1, 0].bar(x, [c/sample_n*100 for c in covered], color=['#2196F3', '#4CAF50'], alpha=0.85, label='Covered')
    axes[1, 0].bar(x, [nc/sample_n*100 for nc in not_covered], bottom=[c/sample_n*100 for c in covered],
                   color='#e0e0e0', alpha=0.85, label='Not covered')
    for i, c in enumerate(covered):
        axes[1, 0].text(i, c/sample_n*100/2, f'{c/sample_n:.0%}', ha='center', va='center',
                       fontweight='bold', color='white')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(methods)
    axes[1, 0].set_ylabel('% of narratives')
    axes[1, 0].set_ylim(0, 115)
    axes[1, 0].set_title('Coverage Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, axis='y', alpha=0.3)

    # Direction breakdown (rule-based only)
    direction_counts = Counter(t['direction'] for t in rule_triples)
    dir_labels = list(direction_counts.keys())
    dir_vals = list(direction_counts.values())
    colors_dir = ['#2196F3' if d == 'forward' else '#FF9800' for d in dir_labels]
    axes[1, 1].pie(dir_vals, labels=dir_labels, colors=colors_dir, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Rule-based: Causal Direction')

    plt.suptitle('Traditional NLP - Rule-based & Dep-parse Overview', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, plots_dir, 'eval_traditional_nlp.png')


def plot_llm_analysis(
    llm_triples: List[dict],
    sample_n: int,
    plots_dir: Path,
):
    """Overview of LLM extraction."""
    if not llm_triples:
        return

    pattern_counts = Counter(t['relation'] for t in llm_triples)
    per_ev = Counter(t['ev_id'] for t in llm_triples)
    densities = list(per_ev.values())
    ev_with = len(per_ev)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Top relations
    top_rels = sorted(pattern_counts.items(), key=lambda x: -x[1])[:15]
    rel_labels = [r for r, _ in top_rels]
    rel_vals = [v for _, v in top_rels]
    neg_kw = ('not', 'fail', 'preclu', 'prevent')
    colors_rel = ['#e74c3c' if any(kw in r.lower() for kw in neg_kw) else '#9C27B0' for r in rel_labels]
    bars = axes[0].barh(rel_labels, rel_vals, color=colors_rel, alpha=0.85)
    for bar, val in zip(bars, rel_vals):
        axes[0].text(val + 5, bar.get_y() + bar.get_height()/2, str(val), va='center', fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Count')
    axes[0].set_title('LLM: Top Relation Phrases')
    axes[0].grid(True, axis='x', alpha=0.3)

    # Density
    axes[1].hist(densities, bins=min(20, max(densities)), color='#9C27B0', edgecolor='white', alpha=0.85)
    axes[1].axvline(np.mean(densities), color='red', linestyle='--', linewidth=1.5, label=f'Mean={np.mean(densities):.1f}')
    axes[1].set_xlabel('Triples per narrative')
    axes[1].set_ylabel('Count')
    axes[1].set_title('LLM: Density Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Coverage donut
    not_covered = max(0, sample_n - ev_with)
    axes[2].pie([ev_with, not_covered], labels=['Covered', 'Not covered'], colors=['#9C27B0', '#e0e0e0'],
               autopct='%1.1f%%', startangle=90)
    axes[2].set_title(f'LLM: Coverage\n{ev_with}/{sample_n}')

    plt.suptitle('LLM (Mistral-7B) - Causal Extraction Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, plots_dir, 'eval_llm_analysis.png')


# =========================================================================
# Knowledge Graph
# =========================================================================

def plot_kg_stats(stats_rules: dict, stats_deps: dict, stats_all: dict, plots_dir: Path):
    """Plot knowledge graph statistics."""
    sources = ['Rule-based', 'Dep-parse', 'All Combined']
    stats = [stats_rules, stats_deps, stats_all]
    colors = ['#2196F3', '#4CAF50', '#9C27B0']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Nodes
    nodes = [s.get('num_nodes', 0) for s in stats]
    axes[0, 0].bar(sources, nodes, color=colors, alpha=0.85, edgecolor='white')
    for i, val in enumerate(nodes):
        axes[0, 0].text(i, val + 500, f'{val:,}', ha='center', va='bottom', fontweight='bold')
    axes[0, 0].set_ylabel('Number of Nodes')
    axes[0, 0].set_title('Knowledge Graph - Nodes')
    axes[0, 0].grid(True, axis='y', alpha=0.3)

    # Edges
    edges = [s.get('num_edges', 0) for s in stats]
    axes[0, 1].bar(sources, edges, color=colors, alpha=0.85, edgecolor='white')
    for i, val in enumerate(edges):
        axes[0, 1].text(i, val + 200, f'{val:,}', ha='center', va='bottom', fontweight='bold')
    axes[0, 1].set_ylabel('Number of Edges')
    axes[0, 1].set_title('Knowledge Graph - Edges')
    axes[0, 1].grid(True, axis='y', alpha=0.3)

    # Weakly Connected Components
    wccs = [s.get('weakly_connected_components', 0) for s in stats]
    axes[1, 0].bar(sources, wccs, color=colors, alpha=0.85, edgecolor='white')
    for i, val in enumerate(wccs):
        axes[1, 0].text(i, val + 100, f'{val:,}', ha='center', va='bottom', fontweight='bold')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Knowledge Graph - Weakly Connected Components')
    axes[1, 0].grid(True, axis='y', alpha=0.3)

    # Density
    densities = [s.get('density', 0) for s in stats]
    axes[1, 1].bar(sources, densities, color=colors, alpha=0.85, edgecolor='white')
    for i, val in enumerate(densities):
        axes[1, 1].text(i, val + val*0.1, f'{val:.2e}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Knowledge Graph - Density')
    axes[1, 1].grid(True, axis='y', alpha=0.3)

    plt.suptitle('Knowledge Graph Statistics', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, plots_dir, 'eval_kg_stats.png')
