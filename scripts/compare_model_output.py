import glob
import os
import re
from collections import Counter
import pandas as pd
from readability import Readability
from readability.exceptions import ReadabilityException
from logodds import compute_log_odds, top_words

# ---------------------------------------------------------
# Dynamic Path Setup
# ---------------------------------------------------------
# SCRIPT_DIR is the 'scripts/' directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT is the parent folder containing responses_nfp/ and scripts/
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "responses_nfp")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def clean_text_list(series: pd.Series) -> list[str]:
    """Drops null and whitespace-only strings from a Series."""
    return [str(t).strip() for t in series.dropna() if str(t).strip()]

def get_word_counts(texts: list[str]) -> dict[str, int]:
    """
    Extracts lowercase alphabetical tokens from all texts and returns
    a dictionary mapping words to their frequency counts.
    """
    words = []
    for text in texts:
        words.extend(re.findall(r"\b[a-zA-Z]{2,}\b", text.lower()))
    return dict(Counter(words))

def compute_avg_word_length(texts: list[str]) -> float:
    """Computes average character length per word across texts."""
    words = [w for text in texts for w in re.findall(r"\b\w+\b", text)]
    return (sum(len(w) for w in words) / len(words)) if words else 0.0

def compute_flesch_kincaid_metrics(text: str) -> tuple[float | None, str | None]:
    """
    Computes both the numeric score and grade_level label via py-readability-metrics.
    Returns (fk.score, fk.grade_level). Requires >= 100 words.
    """
    try:
        r = Readability(text)
        fk = r.flesch_kincaid()
        return fk.score, fk.grade_level
    except (ReadabilityException, ValueError, ZeroDivisionError):
        return None, None

def compute_corpus_fk_metrics(texts: list[str]) -> tuple[float | None, str | None]:
    """
    Evaluates Flesch-Kincaid across individual responses having >= 100 words,
    returning the average score and modal (most frequent) grade_level.
    Falls back to evaluating the chained corpus if single responses are short.
    """
    scores = []
    grade_levels = []

    for text in texts:
        score, grade_level = compute_flesch_kincaid_metrics(text)
        if score is not None:
            scores.append(score)
            grade_levels.append(grade_level)

    if scores:
        mean_score = sum(scores) / len(scores)
        modal_grade = Counter(grade_levels).most_common(1)[0][0]
        return mean_score, modal_grade

    chained_text = " ".join(texts)
    return compute_flesch_kincaid_metrics(chained_text)


# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

def main():
    baseline_filename = "responses_explicit_nfp_baseline.csv"
    baseline_path = os.path.join(DATA_DIR, baseline_filename)

    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline file not found at: {baseline_path}")

    # 1. Process Baseline
    df_baseline = pd.read_csv(baseline_path)
    baseline_texts = clean_text_list(df_baseline["model_output"])
    baseline_counts = get_word_counts(baseline_texts)  # Corpus 2
    
    baseline_avg_len = compute_avg_word_length(baseline_texts)
    baseline_fk_score, baseline_fk_grade = compute_corpus_fk_metrics(baseline_texts)

    print("=" * 80)
    print(f"BASELINE METRICS ({len(baseline_texts)} responses, {sum(baseline_counts.values())} total tokens)")
    print(f"  - Avg Word Length: {baseline_avg_len:.3f} chars")
    print(f"  - FK Score:        {baseline_fk_score:.2f}" if baseline_fk_score else "  - FK Score:        N/A")
    print(f"  - FK Grade Level:  {baseline_fk_grade if baseline_fk_grade else 'N/A'}")
    print("=" * 80 + "\n")

    # 2. Collect Dialect Files from responses_nfp/
    dialect_files = sorted(
        [
            f for f in glob.glob(os.path.join(DATA_DIR, "responses_explicit_nfp_*.csv"))
            if os.path.basename(f) != baseline_filename
        ]
    )

    if not dialect_files:
        print(f"No dialect files found matching 'responses_explicit_nfp_*.csv' in {DATA_DIR}")
        return

    summary_rows = []
    log_odds_results = {}

    for file_path in dialect_files:
        dialect_name = (
            os.path.basename(file_path)
            .replace("responses_explicit_nfp_", "")
            .replace(".csv", "")
        )
        df_dialect = pd.read_csv(file_path)
        dialect_texts = clean_text_list(df_dialect["model_output"])
        dialect_counts = get_word_counts(dialect_texts)  # Corpus 1

        # Text readability metrics
        avg_len = compute_avg_word_length(dialect_texts)
        fk_score, fk_grade = compute_corpus_fk_metrics(dialect_texts)

        # 3. Log-Odds with Dirichlet Prior
        # Corpus 1 = dialect, Corpus 2 = baseline
        try:
            scores = compute_log_odds(dialect_counts, baseline_counts)
            
            top_dialect_words = top_words(scores, n=20)
            top_baseline_words = top_words(scores, n=20, corpus=2)
            
            log_odds_results[dialect_name] = {
                "dialect_top": top_dialect_words,
                "baseline_top": top_baseline_words,
            }
        except Exception as e:
            log_odds_results[dialect_name] = {"error": str(e)}

        delta_score = (
            round(fk_score - baseline_fk_score, 2)
            if (fk_score is not None and baseline_fk_score is not None)
            else None
        )

        summary_rows.append({
            "Dialect": dialect_name,
            "Num Responses": len(dialect_texts),
            "Avg Word Length": round(avg_len, 3),
            "Delta Word Length": round(avg_len - baseline_avg_len, 3),
            "FK Score": round(fk_score, 2) if fk_score is not None else None,
            "Delta FK Score": delta_score,
            "FK Grade Level": fk_grade,
        })

    # 4. Print Summary Table
    summary_df = pd.DataFrame(summary_rows)
    print("### Summary Comparison Table ###")
    print(summary_df.to_string(index=False))
    print("\n" + "=" * 80)

    # 5. Print Top Distinctive Words (Log-Odds)
    print("### Distinctive Words (Log-Odds with Dirichlet Prior) ###")
    for dialect, res in log_odds_results.items():
        print(f"\n--- {dialect.upper()} vs. BASELINE ---")
        if "error" in res:
            print(f"  Error: {res['error']}")
        else:
            print(f"  Top words characteristic of {dialect}:")
            print(f"    {res['dialect_top']}")
            print("  Top words characteristic of Baseline:")
            print(f"    {res['baseline_top']}")

    # Save metrics to results/
    output_path = os.path.join(OUTPUT_DIR, "dialect_baseline_comparison_metrics.csv")
    summary_df.to_csv(output_path, index=False)
    print(f"\nReadability metrics saved to '{output_path}'.")


if __name__ == "__main__":
    main()