"""
Exploratory data analysis for NTSB aviation accident narratives.
"""
import re
import os
from collections import Counter
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

# Matplotlib import with non-interactive backend fallback
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'shall', 'can', 'not',
    'that', 'this', 'these', 'those', 'it', 'its', 'he', 'she', 'they',
    'them', 'their', 'his', 'her', 'our', 'we', 'you', 'your', 'i', 'my',
    'me', 'us', 'who', 'which', 'what', 'when', 'where', 'how', 'if',
    'then', 'than', 'so', 'no', 'also', 'into', 'up', 'out', 'about',
    'after', 'before', 'during', 'while', 'over', 'under', 'between',
    'through', 'there', 'here', 'all', 'any', 'each', 'both', 'such',
    'other', 'same', 'own', 'just', 'more', 'most', 'very', 'too',
}

CAUSAL_KEYWORDS = [
    "due to",
    "caused by",
    "resulted in",
    "led to",
    "because of",
    "as a result of",
    "attributed to",
    "contributed to",
    "resulting from",
]


def basic_stats(df: pd.DataFrame) -> dict:
    """Return basic corpus statistics."""
    return {
        'total_records': len(df),
        'unique_events': df['ev_id'].nunique() if 'ev_id' in df.columns else None,
        'records_with_narrative': int(df['narr_accf'].notna().sum()) if 'narr_accf' in df.columns else None,
        'records_with_findings': int(df['finding_description'].notna().sum()) if 'finding_description' in df.columns else None,
        'avg_narrative_length': round(df['text_length'].mean(), 1) if 'text_length' in df.columns else None,
        'median_narrative_length': round(df['text_length'].median(), 1) if 'text_length' in df.columns else None,
        'avg_sentence_count': round(df['sentence_count'].mean(), 2) if 'sentence_count' in df.columns else None,
    }


def category_distribution(df: pd.DataFrame) -> pd.Series:
    """Return value counts of top_category."""
    if 'top_category' not in df.columns:
        raise ValueError("DataFrame missing 'top_category' column — run preprocess_data first.")
    return df['top_category'].value_counts()


def top_words(texts: List[str], top_n: int = 50) -> dict:
    """
    Compute word frequency dict from a list of texts,
    excluding common stopwords and words shorter than 3 characters.
    """
    counter = Counter()
    word_re = re.compile(r"[a-z]+(?:'[a-z]+)?")
    for text in texts:
        if not isinstance(text, str):
            continue
        words = word_re.findall(text.lower())
        words = [w for w in words if len(w) >= 3 and w not in STOPWORDS]
        counter.update(words)
    return dict(counter.most_common(top_n))


def causal_keyword_frequency(texts: List[str]) -> dict:
    """
    Count occurrences of each causal keyword phrase across all texts.
    """
    counts = {kw: 0 for kw in CAUSAL_KEYWORDS}
    for text in texts:
        if not isinstance(text, str):
            continue
        text_lower = text.lower()
        for kw in CAUSAL_KEYWORDS:
            counts[kw] += text_lower.count(kw)
    return counts


def plot_category_distribution(df: pd.DataFrame, save_path: Optional[str] = None):
    """Bar chart of top_category distribution."""
    if not PLOTTING_AVAILABLE:
        print("[EDA] matplotlib/seaborn not available — skipping plot.")
        return

    dist = category_distribution(df)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=dist.values, y=dist.index, ax=ax, palette='viridis')
    ax.set_title('Finding Category Distribution', fontsize=14)
    ax.set_xlabel('Count')
    ax.set_ylabel('Category')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[EDA] Saved category distribution plot to {save_path}")
    plt.close(fig)


def plot_narrative_lengths(df: pd.DataFrame, save_path: Optional[str] = None):
    """Histogram of narrative text lengths with mean and median lines."""
    if not PLOTTING_AVAILABLE:
        print("[EDA] matplotlib/seaborn not available — skipping plot.")
        return

    if 'text_length' not in df.columns:
        print("[EDA] 'text_length' column missing — skipping plot.")
        return

    lengths = df['text_length'].dropna()
    mean_len = lengths.mean()
    median_len = lengths.median()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lengths, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(mean_len, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_len:.0f}')
    ax.axvline(median_len, color='orange', linestyle='--', linewidth=1.5, label=f'Median: {median_len:.0f}')
    ax.set_title('Narrative Text Length Distribution', fontsize=14)
    ax.set_xlabel('Character Count')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[EDA] Saved narrative length plot to {save_path}")
    plt.close(fig)


def plot_causal_keywords(texts: List[str], save_path: Optional[str] = None):
    """Bar chart of causal keyword frequencies across the corpus."""
    if not PLOTTING_AVAILABLE:
        print("[EDA] matplotlib/seaborn not available — skipping plot.")
        return

    freq = causal_keyword_frequency(texts)
    keywords = list(freq.keys())
    counts = list(freq.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=counts, y=keywords, ax=ax, palette='rocket')
    ax.set_title('Causal Keyword Frequency in Narratives', fontsize=14)
    ax.set_xlabel('Occurrences')
    ax.set_ylabel('Keyword')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[EDA] Saved causal keyword plot to {save_path}")
    plt.close(fig)


def run_eda(df: pd.DataFrame, output_dir: Union[str, Path]) -> dict:
    """
    Run all EDA steps: print results to stdout, save plots, return summary dict.
    """
    output_dir = Path(output_dir)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # Basic stats
    stats = basic_stats(df)
    print("\n--- Basic Statistics ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Category distribution
    print("\n--- Category Distribution ---")
    try:
        dist = category_distribution(df)
        print(dist.to_string())
    except ValueError as e:
        print(f"  (skipped: {e})")
        dist = pd.Series(dtype=int)

    # Top words
    print("\n--- Top 20 Words ---")
    texts = df['narr_clean'].tolist() if 'narr_clean' in df.columns else df['narr_accf'].astype(str).tolist()
    words = top_words(texts, top_n=20)
    for word, count in list(words.items())[:20]:
        print(f"  {word}: {count}")

    # Causal keyword frequency
    print("\n--- Causal Keyword Frequencies ---")
    kw_freq = causal_keyword_frequency(texts)
    for kw, count in sorted(kw_freq.items(), key=lambda x: -x[1]):
        print(f"  '{kw}': {count}")

    # Plots
    print("\n--- Generating Plots ---")
    plot_category_distribution(df, save_path=str(plots_dir / 'category_distribution.png'))
    plot_narrative_lengths(df, save_path=str(plots_dir / 'narrative_lengths.png'))
    plot_causal_keywords(texts, save_path=str(plots_dir / 'causal_keywords.png'))

    summary = {
        'stats': stats,
        'category_distribution': dist.to_dict(),
        'top_words': words,
        'causal_keyword_frequency': kw_freq,
    }
    return summary
