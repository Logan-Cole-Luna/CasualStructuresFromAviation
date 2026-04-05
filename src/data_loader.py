"""
Data loading and preprocessing for NTSB aviation accident narratives.
"""
import re
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV dataset, drop rows where narr_accf is null or empty, return DataFrame.
    """
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=['narr_accf'])
    df = df[df['narr_accf'].astype(str).str.strip() != '']
    df = df.reset_index(drop=True)
    return df


def parse_finding(finding_str: str) -> dict:
    """
    Parse a finding_description string into structured fields.

    Format: "Category-Subcategory-Detail1-Detail2-Actor - C"
    Returns dict with keys: category, subcategory, detail1, detail2, actor, cause_type.
    """
    if not isinstance(finding_str, str) or not finding_str.strip():
        return {
            'category': None,
            'subcategory': None,
            'detail1': None,
            'detail2': None,
            'actor': None,
            'cause_type': None,
        }

    # Strip trailing cause type: " - C" or " - F"
    cause_type = None
    match = re.search(r'\s*-\s*([CF])\s*$', finding_str)
    if match:
        cause_type = match.group(1)
        finding_str = finding_str[:match.start()]

    # Split remaining string by '-', limit to 5 parts
    parts = finding_str.split('-', 4)
    parts = [p.strip() for p in parts]

    # Pad to 5 elements
    while len(parts) < 5:
        parts.append(None)

    return {
        'category':    parts[0] if parts[0] else None,
        'subcategory': parts[1] if len(parts) > 1 and parts[1] else None,
        'detail1':     parts[2] if len(parts) > 2 and parts[2] else None,
        'detail2':     parts[3] if len(parts) > 3 and parts[3] else None,
        'actor':       parts[4] if len(parts) > 4 and parts[4] else None,
        'cause_type':  cause_type,
    }


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns to the DataFrame:
      - top_category:    first dash-delimited token of finding_description
      - narr_clean:      whitespace-normalized narrative text
      - text_length:     character length of narr_clean
      - sentence_count:  approximate sentence count
      - finding_parsed:  dict from parse_finding
      - cause_type:      'C' or 'F' (from finding_parsed)
    """
    df = df.copy()

    # top_category: first segment of finding_description before any '-'
    def _top_category(s):
        if not isinstance(s, str) or not s.strip():
            return None
        # Strip trailing cause type first
        s = re.sub(r'\s*-\s*[CF]\s*$', '', s)
        return s.split('-')[0].strip()

    df['top_category'] = df['finding_description'].apply(_top_category)

    # narr_clean: collapse all whitespace (newlines, tabs, multiple spaces) to single space
    df['narr_clean'] = (
        df['narr_accf']
        .astype(str)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )

    df['text_length'] = df['narr_clean'].str.len()

    # Approximate sentence count by counting terminal punctuation
    df['sentence_count'] = df['narr_clean'].apply(
        lambda t: max(1, len(re.findall(r'[.!?]+', t)))
    )

    df['finding_parsed'] = df['finding_description'].apply(parse_finding)

    df['cause_type'] = df['finding_parsed'].apply(
        lambda d: d.get('cause_type') if isinstance(d, dict) else None
    )

    return df


def get_label_mapping(df: pd.DataFrame, col: str = 'top_category') -> dict:
    """
    Return a dict mapping each unique label in `col` to an integer index,
    sorted alphabetically for reproducibility.
    """
    labels = sorted(df[col].dropna().unique().tolist())
    return {label: idx for idx, label in enumerate(labels)}
