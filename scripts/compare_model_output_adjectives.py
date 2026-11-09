import glob
import os
import re
import sys
from collections import Counter
import pandas as pd
import spacy
from logodds import compute_log_odds, top_words

# ---------------------------------------------------------
# Dynamic Path Setup & spaCy Initialization
# ---------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

nlp = spacy.load("en_core_web_sm", disable=["ner"])


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def read_outputs(paths, limit=None):
    if isinstance(paths, str):
        paths = [paths]
    texts = []
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            if limit:
                df = df.iloc[:limit]
            texts.extend([str(t).strip() for t in df["model_output"].dropna() if str(t).strip()])
    return texts


def get_adjective_counts(texts, batch_size=50):
    """Extracts lowercase lemmatized adjectives (token.pos_ == 'ADJ') via spaCy, treating 'non-standard' as one word."""
    cleaned_texts = [
        re.sub(r"\bnon[- ]standard\b", "nonstandard", t, flags=re.IGNORECASE)
        for t in texts
    ]
    lemmas = []
    for doc in nlp.pipe(cleaned_texts, batch_size=batch_size):
        for token in doc:
            lemma = token.lemma_.lower()
            if lemma in ("nonstandard", "non-standard") or token.text.lower() in ("nonstandard", "non-standard"):
                lemmas.append("non-standard")
            elif token.pos_ == "ADJ" and token.is_alpha and len(lemma) >= 2:
                lemmas.append(lemma)
    return dict(Counter(lemmas))


def filter_counts(counts, min_count=5):
    return {w: c for w, c in counts.items() if c >= min_count}


# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

def main():
    default_dir = os.path.join(PROJECT_ROOT, "responses_elephant", "qwen_qwen3.5-flash-02-23")
    data_dir = sys.argv[1] if len(sys.argv) > 1 else default_dir
    dataset_arg = sys.argv[2] if len(sys.argv) > 2 else "AITA-NTA-OG"
    baseline_limit = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else None
    model_name = os.path.basename(os.path.normpath(data_dir))

    if dataset_arg.lower() in ["combined", "both", "all", "aita-combined"]:
        target_datasets = ["AITA-YTA", "AITA-NTA-OG"]
        dataset_name = "AITA-COMBINED"
    elif "," in dataset_arg:
        target_datasets = [d.strip().replace(".csv", "") for d in dataset_arg.split(",")]
        dataset_name = "-".join(target_datasets)
    else:
        target_datasets = [dataset_arg.replace(".csv", "")]
        dataset_name = target_datasets[0]

    baseline_paths = [os.path.join(data_dir, "baseline", f"{ds}.csv") for ds in target_datasets]
    if all(os.path.exists(p) for p in baseline_paths):
        dialect_dirs = [d for d in sorted(os.listdir(data_dir)) if d != "baseline" and os.path.isdir(os.path.join(data_dir, d))]
        dialect_files = [(d, [os.path.join(data_dir, d, f"{ds}.csv") for ds in target_datasets]) for d in dialect_dirs]
    else:
        baseline_paths = [os.path.join(data_dir, "responses_explicit_nfp_baseline.csv")]
        dialect_files = [
            (os.path.basename(f).replace("responses_explicit_nfp_", "").replace(".csv", ""), [f])
            for f in sorted(glob.glob(os.path.join(data_dir, "responses_explicit_nfp_*.csv")))
            if os.path.basename(f) != "responses_explicit_nfp_baseline.csv"
        ]
        dataset_name = "cancer_nfp"

    # Baseline adjectives
    baseline_texts = read_outputs(baseline_paths, limit=baseline_limit)
    baseline_counts = filter_counts(get_adjective_counts(baseline_texts), min_count=5)

    print("=" * 80)
    limit_note = f" (capped to {baseline_limit}/dataset)" if baseline_limit else ""
    print(f"BASELINE ADJECTIVES: {model_name} | {dataset_name}{limit_note}")
    print(f"  - Responses:        {len(baseline_texts)}")
    print(f"  - Total Tokens:     {sum(baseline_counts.values())}")
    print(f"  - Unique Lemmas:    {len(baseline_counts)} (count >= 5)")
    print("=" * 80 + "\n")

    summary_rows = []
    log_odds_results = {}

    for dialect, paths in dialect_files:
        texts = read_outputs(paths)
        if not texts:
            continue
        dialect_counts = filter_counts(get_adjective_counts(texts), min_count=5)
        scores = compute_log_odds(dialect_counts, baseline_counts)

        top_dialect = top_words(scores, n=20)
        top_baseline = top_words(scores, n=20, corpus=2)
        log_odds_results[dialect] = (top_dialect, top_baseline)

        summary_rows.append({
            "Dialect": dialect,
            "Num Responses": len(texts),
            "Total Adjectives": sum(dialect_counts.values()),
            "Unique Adjectives": len(dialect_counts),
            "Top Dialect Adjectives": ", ".join([w for w, _ in top_dialect[:5]]),
            "Top Baseline Adjectives": ", ".join([w for w, _ in top_baseline[:5]]),
        })

    summary_df = pd.DataFrame(summary_rows)
    print("### Summary Comparison Table (Adjectives) ###")
    print(summary_df.to_string(index=False))

    print("\n### Distinctive Adjectives (Log-Odds with Dirichlet Prior, min_count >= 5) ###")
    for dialect, (top_dialect, top_baseline) in log_odds_results.items():
        print(f"\n--- {dialect.upper()} vs. BASELINE ---")
        print(f"  Top adjectives characteristic of {dialect}:")
        print(f"    {top_dialect}")
        print("  Top adjectives characteristic of Baseline:")
        print(f"    {top_baseline}")

    output_path = os.path.join(OUTPUT_DIR, f"{model_name}_{dataset_name}_adjectives_metrics.csv")
    summary_df.to_csv(output_path, index=False)
    print(f"\nAdjective metrics saved to '{output_path}'.")


if __name__ == "__main__":
    main()
