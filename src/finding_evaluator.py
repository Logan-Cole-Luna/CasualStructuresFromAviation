"""
finding_evaluator.py — Ground-truth evaluation using NTSB finding_description.

The finding_description column contains NTSB's official causal determinations
in a hierarchical taxonomy:
    Category - Subcategory - Sub-sub - ... - Actor - CauseCode

CauseCode:
    'C' = Contributing cause (this IS a cause)
    'F' = Finding (observed but not designated as a cause)
    (other values appear for equipment-type findings with no actor code)

This module computes three extraction-quality metrics against this ground truth:

1. category_alignment_score
   For each ev_id where we extracted >= 1 triple AND a finding exists:
   Classify the cause text into one of the 4 NTSB top-level categories using
   keyword heuristics, then compare to the official finding's top category.
   Score = % of ev_ids where the predicted category matches the official one.

2. cause_confirmed_coverage
   Denominator: ev_ids that have >= 1 'C'-coded finding (accidents with a
                confirmed cause that the model *should* be able to extract).
   Numerator:   of those, how many did the model actually extract >= 1 triple for.
   This is a stricter and more meaningful coverage metric than raw coverage.

3. finding_keyword_recall
   For each ev_id, tokenize the finding hierarchy (split by '-', drop 'C'/'F'
   codes and short tokens) to build a set of ground-truth concept tokens.
   Compute what % of those tokens appear anywhere in the extracted cause or
   effect text.  Average across ev_ids.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
    SKLEARN_METRICS_AVAILABLE = True
except ImportError:
    SKLEARN_METRICS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Category keyword heuristics
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    'Personnel issues': [
        'pilot', 'crew', 'captain', 'officer', 'student', 'instructor',
        'decision', 'judgment', 'attention', 'situational', 'awareness',
        'fatigue', 'training', 'procedure', 'checklist', 'error',
        'workload', 'distraction', 'scan', 'monitor', 'experience',
        'planning', 'action', 'omission', 'communication', 'coordination',
    ],
    'Aircraft': [
        'engine', 'fuel', 'power', 'propeller', 'rotor', 'blade',
        'gear', 'brake', 'flap', 'control', 'aileron', 'elevator',
        'rudder', 'hydraulic', 'electrical', 'battery', 'circuit',
        'mechanical', 'structural', 'airframe', 'component', 'system',
        'failure', 'malfunction', 'fatigue', 'corrosion', 'wear',
        'carburetor', 'manifold', 'exhaust', 'oil', 'ignition',
        'magneto', 'cylinder', 'piston', 'crankshaft', 'bearing',
    ],
    'Environmental issues': [
        'weather', 'wind', 'gust', 'turbulence', 'icing', 'ice',
        'fog', 'visibility', 'cloud', 'ceiling', 'precipitation',
        'rain', 'snow', 'density', 'altitude', 'terrain', 'obstacle',
        'bird', 'wildlife', 'night', 'dark',
    ],
    'Organizational issues': [
        'maintenance', 'management', 'organization', 'oversight',
        'regulation', 'policy', 'procedure', 'inspection', 'supervision',
        'dispatch', 'scheduling',
    ],
}

# Short tokens to skip when building finding keyword sets
_SKIP_TOKENS = frozenset({
    'c', 'f', 'the', 'a', 'an', 'and', 'or', 'of', 'in', 'to',
    'not', 'by', 'on', 'at', 'for', 'with', 'issues', 'general',
    'other', 'unknown', 'misc', 'attained', 'maintained',
    'use', 'effect', 'type', 'condition', 'related',
})

# Soft matching configuration for finding_description overlap.
_FUZZY_MATCH_THRESHOLD = 0.86
_MIN_DETECTION_COVERAGE = 0.10
_COMPOSITE_F1_WEIGHT = 0.5
_COMPOSITE_COVERAGE_WEIGHT = 0.5

# Lightweight domain synonyms/variants to improve semantic overlap recall.
_TOKEN_SYNONYMS: Dict[str, List[str]] = {
    'pilot': ['aviator', 'captain'],
    'engine': ['powerplant', 'motor'],
    'propeller': ['prop'],
    'airplane': ['aircraft', 'plane'],
    'aircraft': ['airplane', 'plane'],
    'weather': ['meteorological', 'wx'],
    'visibility': ['vis'],
    'fatigue': ['tiredness', 'exhaustion'],
    'failure': ['malfunction', 'breakdown'],
    'malfunction': ['failure', 'fault'],
    'maintenance': ['service', 'inspection'],
    'checklist': ['check list'],
    'decision': ['judgment', 'judgement'],
    'judgment': ['decision', 'judgement'],
    'stall': ['stalled', 'aerodynamicstall'],
    'icing': ['ice'],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize_finding(finding: str) -> List[str]:
    """Split a finding string into meaningful concept tokens."""
    raw = re.split(r'[-/]', finding)
    tokens = []
    for part in raw:
        part = part.strip().lower()
        if len(part) >= 4 and part not in _SKIP_TOKENS:
            # Further split on spaces and keep multi-word tokens as is
            for word in re.split(r'\s+', part):
                word = re.sub(r'[^a-z]', '', word)
                if len(word) >= 4 and word not in _SKIP_TOKENS:
                    tokens.append(word)
    return tokens


def _normalize_token(tok: str) -> str:
    """Lowercase + alpha cleanup + lightweight stemming for soft matching."""
    tok = re.sub(r'[^a-z]', '', tok.lower())
    if len(tok) <= 3:
        return tok

    # Lightweight stemming rules keep implementation dependency-free.
    for suf in ('ization', 'ations', 'ation', 'ments', 'ment', 'ingly', 'edly', 'ness', 'ingly'):
        if tok.endswith(suf) and len(tok) > len(suf) + 3:
            tok = tok[:-len(suf)]
            break
    for suf in ('ing', 'ed', 'ies', 'es', 's'):
        if tok.endswith(suf) and len(tok) > len(suf) + 2:
            if suf == 'ies':
                tok = tok[:-3] + 'y'
            else:
                tok = tok[:-len(suf)]
            break
    return tok


def _tokenize_text_soft(text: str) -> List[str]:
    """Tokenize arbitrary text into normalized tokens for soft overlap scoring."""
    raw = re.findall(r'[a-zA-Z]+', text.lower())
    norm = [_normalize_token(t) for t in raw]
    return [t for t in norm if len(t) >= 3 and t not in _SKIP_TOKENS]


def _expand_synonyms(tok: str) -> List[str]:
    """Return normalized token + synonym variants for soft match lookup."""
    variants = {_normalize_token(tok)}
    for s in _TOKEN_SYNONYMS.get(tok, []):
        variants.add(_normalize_token(s))
    # Also allow reverse synonym mapping.
    for base, syns in _TOKEN_SYNONYMS.items():
        if tok in syns:
            variants.add(_normalize_token(base))
            for s in syns:
                variants.add(_normalize_token(s))
    return [v for v in variants if v]


def _soft_overlap_stats(gt_tokens: List[str], pred_text: str) -> Tuple[int, float]:
    """
    Compute soft overlap of finding tokens against extracted text.

    Matching strategy (in order):
      1) exact normalized token match
      2) synonym variant match
      3) fuzzy match with SequenceMatcher
    """
    if not gt_tokens:
        return 0, 0.0

    pred_tokens = _tokenize_text_soft(pred_text)
    pred_token_set = set(pred_tokens)
    if not pred_tokens:
        return 0, 0.0

    matched = 0
    for tok in gt_tokens:
        n_tok = _normalize_token(tok)
        if not n_tok:
            continue

        if n_tok in pred_token_set:
            matched += 1
            continue

        variants = _expand_synonyms(n_tok)
        if any(v in pred_token_set for v in variants):
            matched += 1
            continue

        # Fuzzy fallback for minor lexical variation (e.g., singular/plural/typo).
        best = 0.0
        for p in pred_token_set:
            if abs(len(p) - len(n_tok)) > 4:
                continue
            best = max(best, SequenceMatcher(None, n_tok, p).ratio())
            if best >= _FUZZY_MATCH_THRESHOLD:
                break
        if best >= _FUZZY_MATCH_THRESHOLD:
            matched += 1

    coverage = matched / max(1, len(gt_tokens))
    return matched, coverage


def _classify_text(text: str) -> str:
    """Heuristic: classify a cause/effect string into an NTSB top category."""
    text_lower = text.lower()
    scores: Dict[str, int] = {}
    for cat, kws in _CATEGORY_KEYWORDS.items():
        scores[cat] = sum(1 for kw in kws if kw in text_lower)
    best_score = max(scores.values())
    if best_score == 0:
        return 'Unknown'
    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def load_findings(data_path: str | Path) -> pd.DataFrame:
    """
    Load finding_description from the dataset and parse into structured columns.

    Returns a DataFrame with columns:
        ev_id, finding_description, category, level1, level2, is_cause
    """
    df = pd.read_csv(data_path, usecols=['ev_id', 'finding_description'])
    df['ev_id'] = df['ev_id'].astype(str)
    df = df.dropna(subset=['finding_description'])

    parts = df['finding_description'].str.split('-')
    df['category'] = parts.str[0].str.strip()
    df['level1']   = parts.str[1].str.strip() if parts.str.len().max() > 1 else ''
    df['level2']   = parts.str[2].str.strip() if parts.str.len().max() > 2 else ''
    df['cause_code'] = parts.str[-1].str.strip()
    df['is_cause'] = df['cause_code'] == 'C'

    return df.reset_index(drop=True)


def evaluate_finding_alignment(
    triples: List[dict],
    findings_df: pd.DataFrame,
    label: str = 'model',
) -> Dict:
    """
    Compute the three finding-alignment metrics for a set of extracted triples.

    Parameters
    ----------
    triples : list of dicts with keys ev_id, cause, effect, relation
    findings_df : output of load_findings()
    label : name of the extraction method (for reporting)

    Returns
    -------
    dict with keys:
        label, total_triples, ev_ids_extracted,
        category_alignment_score,  category_alignment_n,
        cause_confirmed_coverage,  cause_confirmed_n, cause_confirmed_denom,
        finding_keyword_recall,    keyword_recall_n,
        per_category_alignment     (breakdown by NTSB category)
    """
    # Index findings by ev_id
    findings_by_ev: Dict[str, List[pd.Series]] = defaultdict(list)
    for _, row in findings_df.iterrows():
        findings_by_ev[row['ev_id']].append(row)

    # Build per-ev_id triple text (concatenate all causes + effects)
    triple_text_by_ev: Dict[str, str] = defaultdict(str)
    for t in triples:
        eid = str(t.get('ev_id', ''))
        triple_text_by_ev[eid] += ' ' + str(t.get('cause', '')) + ' ' + str(t.get('effect', ''))

    extracted_ev_ids = set(triple_text_by_ev.keys())
    cause_finding_ev_ids = set(
        row['ev_id'] for _, row in findings_df.iterrows() if row['is_cause']
    )

    # ------------------------------------------------------------------
    # Metric 1: Category alignment
    # ------------------------------------------------------------------
    alignment_correct = 0
    alignment_total   = 0
    per_category_correct: Dict[str, int] = defaultdict(int)
    per_category_total:   Dict[str, int] = defaultdict(int)

    for ev_id, text in triple_text_by_ev.items():
        if ev_id not in findings_by_ev:
            continue
        # Use the primary (first) cause finding for comparison
        cause_findings = [r for r in findings_by_ev[ev_id] if r['is_cause']]
        if not cause_findings:
            cause_findings = findings_by_ev[ev_id]   # fall back to all findings
        official_cat = cause_findings[0]['category']

        predicted_cat = _classify_text(text)

        per_category_total[official_cat] += 1
        if predicted_cat == official_cat:
            alignment_correct += 1
            per_category_correct[official_cat] += 1
        alignment_total += 1

    cat_alignment_score = (
        alignment_correct / alignment_total if alignment_total > 0 else 0.0
    )
    per_cat_alignment = {
        cat: {
            'correct': per_category_correct[cat],
            'total':   per_category_total[cat],
            'score':   round(per_category_correct[cat] / max(1, per_category_total[cat]), 4),
        }
        for cat in per_category_total
    }

    # ------------------------------------------------------------------
    # Metric 2: Cause-confirmed coverage
    # ------------------------------------------------------------------
    # Accidents that (a) appear in our dataset AND (b) have a C finding
    eligible_ev_ids = cause_finding_ev_ids & set(findings_by_ev.keys())
    covered_cause_ev_ids = eligible_ev_ids & extracted_ev_ids

    cause_confirmed_cov = (
        len(covered_cause_ev_ids) / len(eligible_ev_ids)
        if eligible_ev_ids else 0.0
    )

    # ------------------------------------------------------------------
    # Metric 3: Finding keyword recall
    # ------------------------------------------------------------------
    recall_scores: List[float] = []
    for ev_id, text in triple_text_by_ev.items():
        if ev_id not in findings_by_ev:
            continue
        # Gather all finding tokens for this accident
        all_tokens: List[str] = []
        for row in findings_by_ev[ev_id]:
            all_tokens.extend(_tokenize_finding(row['finding_description']))
        if not all_tokens:
            continue

        _, coverage = _soft_overlap_stats(all_tokens, text)
        recall_scores.append(coverage)

    avg_keyword_recall = (
        sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    )

    return {
        'label':                     label,
        'total_triples':             len(triples),
        'ev_ids_extracted':          len(extracted_ev_ids),
        'category_alignment_score':  round(cat_alignment_score,  4),
        'category_alignment_n':      alignment_total,
        'cause_confirmed_coverage':  round(cause_confirmed_cov,  4),
        'cause_confirmed_n':         len(covered_cause_ev_ids),
        'cause_confirmed_denom':     len(eligible_ev_ids),
        'finding_keyword_recall':    round(avg_keyword_recall,    4),
        'keyword_recall_n':          len(recall_scores),
        'per_category_alignment':    per_cat_alignment,
    }


def evaluate_classifier_alignment(
    predictions: Dict[str, str],
    findings_df: pd.DataFrame,
    label: str = 'DistilBERT',
) -> Dict:
    """
    Evaluate a classifier (DistilBERT) against finding_description ground truth.

    Parameters
    ----------
    predictions : {ev_id: predicted_category_str}  — one prediction per narrative
    findings_df : output of load_findings()
    label       : display name for the model

    Returns a dict with the same keys as evaluate_finding_alignment so it can
    be included in the same table and plots.  finding_keyword_recall is None
    because a classifier has no free-text output to match against.
    """
    findings_by_ev: Dict[str, List] = defaultdict(list)
    for _, row in findings_df.iterrows():
        findings_by_ev[row['ev_id']].append(row)

    cause_finding_ev_ids = set(
        row['ev_id'] for _, row in findings_df.iterrows() if row['is_cause']
    )

    alignment_correct = 0
    alignment_total   = 0
    per_category_correct: Dict[str, int] = defaultdict(int)
    per_category_total:   Dict[str, int] = defaultdict(int)

    for ev_id, predicted_cat in predictions.items():
        if ev_id not in findings_by_ev:
            continue
        cause_findings = [r for r in findings_by_ev[ev_id] if r['is_cause']]
        official_cat   = (cause_findings or findings_by_ev[ev_id])[0]['category']

        per_category_total[official_cat] += 1
        if predicted_cat == official_cat:
            alignment_correct += 1
            per_category_correct[official_cat] += 1
        alignment_total += 1

    cat_alignment_score = alignment_correct / alignment_total if alignment_total > 0 else 0.0

    eligible  = cause_finding_ev_ids & set(findings_by_ev.keys())
    covered   = eligible & set(predictions.keys())

    per_cat_alignment = {
        cat: {
            'correct': per_category_correct[cat],
            'total':   per_category_total[cat],
            'score':   round(per_category_correct[cat] / max(1, per_category_total[cat]), 4),
        }
        for cat in per_category_total
    }

    return {
        'label':                     label,
        'total_triples':             None,          # N/A — classifier, not extractor
        'ev_ids_extracted':          len(predictions),
        'category_alignment_score':  round(cat_alignment_score, 4),
        'category_alignment_n':      alignment_total,
        'cause_confirmed_coverage':  round(len(covered) / max(1, len(eligible)), 4),
        'cause_confirmed_n':         len(covered),
        'cause_confirmed_denom':     len(eligible),
        'finding_keyword_recall':    None,          # N/A — no free-text output
        'keyword_recall_n':          0,
        'per_category_alignment':    per_cat_alignment,
    }


def print_finding_report(results: List[Dict]) -> None:
    """Pretty-print finding-alignment metrics for multiple models."""
    w = 26
    print(f'\n  {"Metric":<{w}}  ' +
          '  '.join(f'{r["label"]:>18}' for r in results))
    print('  ' + '-' * (w + 22 * len(results)))

    rows = [
        ('ev_ids extracted',        lambda r: f'{r["ev_ids_extracted"]:>18,}'),
        ('Cat. alignment score',    lambda r: f'{r["category_alignment_score"]:>17.1%}'),
        ('  (n ev_ids evaluated)',  lambda r: f'{r["category_alignment_n"]:>18,}'),
        ('Cause-confirmed cov.',    lambda r: f'{r["cause_confirmed_coverage"]:>17.1%}'),
        ('  (n covered / denom)',   lambda r: f'{r["cause_confirmed_n"]}/{r["cause_confirmed_denom"]:>14,}'),
        ('Finding keyword recall',  lambda r: f'{"N/A":>18}' if r["finding_keyword_recall"] is None else f'{r["finding_keyword_recall"]:>17.1%}'),
        ('  (n ev_ids scored)',     lambda r: f'{"N/A":>18}' if r["finding_keyword_recall"] is None else f'{r["keyword_recall_n"]:>18,}'),
    ]

    for label, fmt in rows:
        print(f'  {label:<{w}}  ' + '  '.join(fmt(r) for r in results))

    print()
    # Per-category breakdown
    all_cats = sorted({cat for r in results for cat in r['per_category_alignment']})
    print(f'  Category alignment breakdown:')
    print(f'  {"Category":<28}  ' +
          '  '.join(f'{r["label"]:>18}' for r in results))
    print('  ' + '-' * (30 + 22 * len(results)))
    for cat in all_cats:
        short = cat.replace(' issues', '')
        row_str = f'  {short:<28}  '
        for r in results:
            d = r['per_category_alignment'].get(cat, {})
            if d:
                row_str += f'  {d["score"]:>12.1%} ({d["correct"]}/{d["total"]})'
            else:
                row_str += '                     n/a'
        print(row_str)


def evaluate_detection_metrics(
    triples: List[dict],
    findings_df: pd.DataFrame,
    candidate_ev_ids: List[str],
    label: str = 'model',
) -> Dict:
    """
    Evaluate extraction as binary cause-detection at the narrative level,
    grounded in finding_description content.

    y_true: narrative has >=1 official C-coded finding
    y_pred: model extracted text with meaningful soft overlap to finding tokens
    y_score: fraction of finding tokens covered by extracted text
    """
    candidate_set = set(str(e) for e in candidate_ev_ids)

    cause_ev_ids = set(
        str(row['ev_id'])
        for _, row in findings_df.iterrows()
        if row['is_cause']
    )

    findings_text_by_ev: Dict[str, str] = defaultdict(str)
    findings_tokens_by_ev: Dict[str, set] = defaultdict(set)
    for _, row in findings_df.iterrows():
        eid = str(row['ev_id'])
        if eid not in candidate_set:
            continue
        ftxt = str(row.get('finding_description', '')).strip()
        if not ftxt:
            continue
        findings_text_by_ev[eid] += ' ' + ftxt
        findings_tokens_by_ev[eid].update(_tokenize_finding(ftxt))

    pred_text_by_ev: Dict[str, str] = defaultdict(str)
    for t in triples:
        eid = str(t.get('ev_id', ''))
        if eid in candidate_set:
            pred_text_by_ev[eid] += ' ' + str(t.get('cause', '')) + ' ' + str(t.get('effect', ''))

    y_true: List[int] = []
    y_pred: List[int] = []
    y_score: List[float] = []
    matched_token_counts: List[int] = []
    for eid in sorted(candidate_set):
        truth = 1 if eid in cause_ev_ids else 0

        gt_tokens = list(findings_tokens_by_ev.get(eid, set()))
        pred_text = pred_text_by_ev.get(eid, '')
        match_n, coverage = _soft_overlap_stats(gt_tokens, pred_text)

        # Require non-trivial overlap to count as a positive extraction.
        pred = 1 if (coverage >= _MIN_DETECTION_COVERAGE or match_n >= 2) else 0
        y_true.append(truth)
        y_pred.append(pred)
        y_score.append(float(coverage))
        matched_token_counts.append(match_n)

    if SKLEARN_METRICS_AVAILABLE:
        acc = float(accuracy_score(y_true, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average='binary',
            zero_division=0,
        )
        auc = None
        if len(set(y_true)) > 1:
            try:
                auc = float(roc_auc_score(y_true, y_score))
            except Exception:
                auc = None
    else:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
        acc = (tp + tn) / max(1, tp + tn + fp + fn)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        auc = None

    positives = int(sum(y_true))
    negatives = int(len(y_true) - positives)
    avg_token_coverage = float(sum(y_score) / max(1, len(y_score)))
    composite = (_COMPOSITE_F1_WEIGHT * float(f1)) + (_COMPOSITE_COVERAGE_WEIGHT * avg_token_coverage)

    return {
        'label': label,
        'n_eval': len(y_true),
        'n_true_positive_class': positives,
        'n_true_negative_class': negatives,
        'accuracy': round(acc, 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'f1': round(float(f1), 4),
        'auc_roc': round(float(auc), 4) if auc is not None else None,
        'avg_finding_token_coverage': round(avg_token_coverage, 4),
        'composite_score': round(composite, 4),
        'n_with_any_finding_token_match': int(sum(1 for x in matched_token_counts if x > 0)),
    }


def print_detection_report(results: List[Dict]) -> None:
    """Pretty-print binary cause-detection metrics for multiple models."""
    if not results:
        return

    w = 26
    print(f'\n  {"Metric":<{w}}  ' +
          '  '.join(f'{r["label"]:>18}' for r in results))
    print('  ' + '-' * (w + 22 * len(results)))

    rows = [
        ('n evaluated',             lambda r: f'{r["n_eval"]:>18,}'),
        ('accuracy',                lambda r: f'{r["accuracy"]:>17.1%}'),
        ('precision',               lambda r: f'{r["precision"]:>17.1%}'),
        ('recall',                  lambda r: f'{r["recall"]:>17.1%}'),
        ('f1',                      lambda r: f'{r["f1"]:>17.1%}'),
        ('auc_roc',                 lambda r: f'{"N/A":>18}' if r["auc_roc"] is None else f'{r["auc_roc"]:>17.4f}'),
        ('avg finding-token cover', lambda r: f'{r["avg_finding_token_coverage"]:>17.1%}'),
        ('composite score',         lambda r: f'{r["composite_score"]:>17.1%}'),
        ('n with token match',      lambda r: f'{r["n_with_any_finding_token_match"]:>18,}'),
        ('true class balance (+/-)', lambda r: f'{r["n_true_positive_class"]}/{r["n_true_negative_class"]:>12,}'),
    ]

    for label, fmt in rows:
        print(f'  {label:<{w}}  ' + '  '.join(fmt(r) for r in results))
