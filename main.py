"""
NTSB Aviation Accident Causal Chain Extraction — Baseline
Team: Madeline Gorman, Katherine Hoffsetz, Logan Luna, Stephanie Ramsey
"""
import warnings; warnings.filterwarnings('ignore')
import json
from pathlib import Path

from src.data_loader import load_data, preprocess_data
from src.eda import run_eda
from src.traditional_nlp import batch_extract, load_nlp
from src.knowledge_graph import build_graph, graph_stats, to_neo4j_cypher, visualize_subgraph
from src.transformer_classifier import setup_demo

DATA_PATH = "data/clean/cleaned_narritives_and_findings.csv"
OUTPUT_DIR = Path("outputs")


def main():
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "plots").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "extractions").mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  NTSB Aviation Accident Causal Chain Extraction — Baseline Pipeline")
    print("  Team: Madeline Gorman, Katherine Hoffsetz, Logan Luna, Stephanie Ramsey")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Step 1: Load and preprocess data
    # -----------------------------------------------------------------------
    print("\n[Step 1] Loading and preprocessing data...")
    try:
        df = load_data(DATA_PATH)
        df = preprocess_data(df)
        print(f"  Dataset shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Top categories: {df['top_category'].value_counts().head(5).to_dict()}")
    except Exception as e:
        print(f"  ERROR in data loading: {e}")
        raise  # Data loading is fatal — can't continue without it

    # -----------------------------------------------------------------------
    # Step 2: EDA
    # -----------------------------------------------------------------------
    print("\n[Step 2] Running exploratory data analysis...")
    try:
        eda_summary = run_eda(df, output_dir=OUTPUT_DIR)
        eda_path = OUTPUT_DIR / "eda_summary.json"
        with open(eda_path, "w") as f:
            # Convert any non-serialisable values (e.g. numpy int64) to native Python
            def _default(obj):
                if hasattr(obj, 'item'):
                    return obj.item()
                return str(obj)
            json.dump(eda_summary, f, indent=2, default=_default)
        print(f"  EDA summary saved to {eda_path}")
    except Exception as e:
        print(f"  ERROR in EDA: {e}")

    # -----------------------------------------------------------------------
    # Step 3: Rule-based NLP extraction
    # -----------------------------------------------------------------------
    print("\n[Step 3] Running rule-based causal extraction (1000-sample subset)...")
    all_triples = []
    try:
        sample_size = min(1000, len(df))
        rule_triples = batch_extract(df, nlp=None, sample_n=sample_size)
        all_triples.extend(rule_triples)
        print(f"  Extracted {len(rule_triples)} causal triples from {sample_size} narratives")

        rules_path = OUTPUT_DIR / "extractions" / "causal_triples_rules.json"
        with open(rules_path, "w") as f:
            json.dump(rule_triples, f, indent=2)
        print(f"  Saved to {rules_path}")
    except Exception as e:
        print(f"  ERROR in rule-based extraction: {e}")

    # -----------------------------------------------------------------------
    # Step 4: spaCy dependency parsing extraction (200-sample subset)
    # -----------------------------------------------------------------------
    print("\n[Step 4] Running spaCy dependency parsing extraction (200-sample subset)...")
    dep_triples = []
    try:
        nlp = load_nlp()
        if nlp is not None:
            dep_sample = min(200, len(df))
            dep_triples = batch_extract(df, nlp=nlp, sample_n=dep_sample)
            print(f"  Extracted {len(dep_triples)} triples via dependency parsing from {dep_sample} narratives")

            deps_path = OUTPUT_DIR / "extractions" / "causal_triples_deps.json"
            with open(deps_path, "w") as f:
                json.dump(dep_triples, f, indent=2)
            print(f"  Saved to {deps_path}")
            all_triples.extend(dep_triples)
        else:
            print("  spaCy model not available — skipping dependency parsing.")
            print("  To enable: pip install spacy && python -m spacy download en_core_web_sm")
    except Exception as e:
        print(f"  ERROR in dependency parsing: {e}")

    # -----------------------------------------------------------------------
    # Step 5: Knowledge graph
    # -----------------------------------------------------------------------
    print("\n[Step 5] Building knowledge graph from extracted triples...")
    try:
        if not all_triples:
            print("  No triples extracted — knowledge graph will be empty.")

        G = build_graph(all_triples)
        stats = graph_stats(G)

        print(f"  Graph nodes:  {stats['num_nodes']}")
        print(f"  Graph edges:  {stats['num_edges']}")
        print(f"  Density:      {stats['density']}")
        print(f"  Weakly connected components: {stats['weakly_connected_components']}")
        print(f"  Top causes:   {stats['top_causes'][:5]}")
        print(f"  Top effects:  {stats['top_effects'][:5]}")
        print(f"  Top by betweenness: {stats['top_nodes_by_betweenness']}")

        cypher_path = str(OUTPUT_DIR / "extractions" / "neo4j_import.cypher")
        to_neo4j_cypher(all_triples, path=cypher_path)

        kg_plot_path = str(OUTPUT_DIR / "plots" / "knowledge_graph.png")
        visualize_subgraph(G, top_n=30, save_path=kg_plot_path)

        # Save graph stats as JSON
        stats_path = OUTPUT_DIR / "extractions" / "graph_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Graph stats saved to {stats_path}")

    except Exception as e:
        print(f"  ERROR in knowledge graph: {e}")

    # -----------------------------------------------------------------------
    # Step 6: Transformer classifier setup demo
    # -----------------------------------------------------------------------
    print("\n[Step 6] Setting up transformer classifier demo...")
    try:
        clf = setup_demo(df)
        if clf is not None:
            print("  Classifier initialized successfully.")
    except Exception as e:
        print(f"  ERROR in transformer setup: {e}")

    # -----------------------------------------------------------------------
    # Completion summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Pipeline complete. Output files:")
    print(f"    EDA plots:          {OUTPUT_DIR / 'plots'}/")
    print(f"    EDA summary:        {OUTPUT_DIR / 'eda_summary.json'}")
    print(f"    Rule triples:       {OUTPUT_DIR / 'extractions' / 'causal_triples_rules.json'}")
    print(f"    Dep triples:        {OUTPUT_DIR / 'extractions' / 'causal_triples_deps.json'}")
    print(f"    Neo4j Cypher:       {OUTPUT_DIR / 'extractions' / 'neo4j_import.cypher'}")
    print(f"    KG visualization:   {OUTPUT_DIR / 'plots' / 'knowledge_graph.png'}")
    print(f"    Graph stats:        {OUTPUT_DIR / 'extractions' / 'graph_stats.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
