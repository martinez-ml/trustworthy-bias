import glob
import os
import re
import sys
from collections import Counter

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from readability import Readability
    from readability.exceptions import ReadabilityException
except ImportError:
    Readability = None
    ReadabilityException = Exception

try:
    from logodds import compute_log_odds, top_words
except ImportError:
    import math

    def compute_log_odds(counts1, counts2):
        vocab = set(counts1) | set(counts2)
        prior = {w: float(counts1.get(w, 0) + counts2.get(w, 0)) for w in vocab}
        a0, n1, n2 = sum(prior.values()) or 1.0, sum(counts1.values()) or 1.0, sum(counts2.values()) or 1.0
        scores = {}
        for w in vocab:
            y1, y2, aw = counts1.get(w, 0), counts2.get(w, 0), prior.get(w, 0.0) or 0.01
            delta = math.log((y1 + aw) / (n1 + a0 - y1 - aw)) - math.log((y2 + aw) / (n2 + a0 - y2 - aw))
            scores[w] = delta / math.sqrt(1.0 / (y1 + aw) + 1.0 / (y2 + aw))
        return scores

    def top_words(scores, n=20, corpus=1):
        return sorted(scores.items(), key=lambda x: x[1], reverse=(corpus == 1))[:n]



# ---------------------------------------------------------
# Dynamic Path Setup
# ---------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def clean_text_list(series) -> list[str]:
    """Drops null and whitespace-only strings from an iterable."""
    return [str(t).strip() for t in series if str(t).strip() and str(t) != "nan"]

def read_csv_outputs(paths: list[str] | str, limit_per_file: int | None = None) -> list[str]:
    """Reads model_output column from one or more CSV files, with optional per-file row limit."""
    if isinstance(paths, str):
        paths = [paths]
    all_texts = []
    for path in paths:
        if not os.path.exists(path):
            continue
        file_texts = []
        if pd is not None:
            try:
                df = pd.read_csv(path)
                if limit_per_file:
                    df = df.iloc[:limit_per_file]
                file_texts = clean_text_list(df["model_output"])
            except Exception:
                file_texts = []
        if not file_texts:
            import csv
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    txt = row.get("model_output", "").strip()
                    if txt:
                        file_texts.append(txt)
                        count += 1
                        if limit_per_file and count >= limit_per_file:
                            break
        all_texts.extend(file_texts)
    return all_texts

def get_word_counts(texts: list[str]) -> dict[str, int]:
    """
    Extracts lowercase alphabetical tokens from all texts and returns
    a dictionary mapping words to their frequency counts. Treats 'non-standard' as one word.
    """
    words = []
    for text in texts:
        cleaned = re.sub(r"\bnon[- ]standard\b", "non-standard", text.lower())
        for w in re.findall(r"\b[a-zA-Z]+(?:-[a-zA-Z]+)*\b", cleaned):
            if w == "non-standard" or (w.isalpha() and len(w) >= 2):
                words.append(w)
    return dict(Counter(words))

def filter_word_counts(counts: dict[str, int], min_count: int = 5) -> dict[str, int]:
    """Filters words with frequency >= min_count."""
    return {w: c for w, c in counts.items() if c >= min_count}

def compute_avg_word_length(texts: list[str]) -> float:
    """Computes average character length per word across texts."""
    words = [w for text in texts for w in re.findall(r"\b\w+\b", text)]
    return (sum(len(w) for w in words) / len(words)) if words else 0.0

def compute_flesch_kincaid_metrics(text: str) -> tuple[float | None, str | None]:
    """
    Computes both numeric score and grade_level label via py-readability-metrics.
    Returns (fk.score, fk.grade_level). Requires >= 100 words.
    """
    if Readability is None:
        return None, None
    try:
        r = Readability(text)
        fk = r.flesch_kincaid()
        return fk.score, fk.grade_level
    except (ReadabilityException, ValueError, ZeroDivisionError, Exception):
        return None, None

def compute_corpus_fk_metrics(texts: list[str]) -> tuple[float | None, str | None]:
    """
    Evaluates Flesch-Kincaid across individual responses having >= 100 words,
    returning the average score and modal (most frequent) grade_level.
    Falls back to evaluating the chained corpus if single responses are short.
    """
    if Readability is None:
        return None, None

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
    # Usage: python compare_model_output.py [data_dir] [dataset]
    # e.g.: python compare_model_output.py responses_elephant/qwen_qwen3.5-flash-02-23 AITA-NTA-OG
    default_dir = os.path.join(PROJECT_ROOT, "responses_elephant", "qwen_qwen3.5-flash-02-23")
    data_dir = sys.argv[1] if len(sys.argv) > 1 else default_dir
    dataset_arg = sys.argv[2] if len(sys.argv) > 2 else "AITA-NTA-OG"
    model_name = os.path.basename(os.path.normpath(data_dir))

    # Optional row limit for baseline (e.g. 3rd arg '200' or flag '--limit 200')
    baseline_limit = None
    if os.environ.get("BASELINE_LIMIT"):
        try:
            baseline_limit = int(os.environ["BASELINE_LIMIT"])
        except ValueError:
            pass

    for i, arg in enumerate(sys.argv):
        if arg in ["--baseline_limit", "--limit", "--baseline-limit"] and i + 1 < len(sys.argv):
            try:
                baseline_limit = int(sys.argv[i + 1])
            except ValueError:
                pass
        elif arg.startswith(("--baseline_limit=", "--limit=", "--baseline-limit=")):
            try:
                baseline_limit = int(arg.split("=")[1])
            except ValueError:
                pass

    if baseline_limit is None and len(sys.argv) > 3:
        try:
            baseline_limit = int(sys.argv[3])
        except ValueError:
            pass

    if dataset_arg.lower() in ["combined", "both", "all", "aita-combined"]:
        target_datasets = ["AITA-YTA", "AITA-NTA-OG"]
        dataset_name = "AITA-COMBINED"
    elif "," in dataset_arg:
        target_datasets = [d.strip().replace(".csv", "") for d in dataset_arg.split(",")]
        dataset_name = "-".join(target_datasets)
    else:
        target_datasets = [dataset_arg.replace(".csv", "")]
        dataset_name = target_datasets[0]

    # Support folder-per-dialect (elephant) and fallback for flat files (nfp)
    baseline_paths = [os.path.join(data_dir, "baseline", f"{ds}.csv") for ds in target_datasets]
    if all(os.path.exists(p) for p in baseline_paths):
        dialect_dirs = [d for d in sorted(os.listdir(data_dir)) if d != "baseline" and os.path.isdir(os.path.join(data_dir, d))]
        dialect_files = [
            (d, [os.path.join(data_dir, d, f"{ds}.csv") for ds in target_datasets])
            for d in dialect_dirs
        ]
    else:
        # Fallback for flat directory structure (e.g. responses_nfp)
        baseline_filename = "responses_explicit_nfp_baseline.csv"
        baseline_paths = [os.path.join(data_dir, baseline_filename)]
        if not os.path.exists(baseline_paths[0]):
            raise FileNotFoundError(f"Baseline file not found at: {baseline_paths[0]}")
        dialect_files = [
            (os.path.basename(f).replace("responses_explicit_nfp_", "").replace(".csv", ""), [f])
            for f in sorted(glob.glob(os.path.join(data_dir, "responses_explicit_nfp_*.csv")))
            if os.path.basename(f) != baseline_filename
        ]
        dataset_name = "cancer_nfp"

    # 1. Process Baseline (apply limit if specified)
    baseline_texts = read_csv_outputs(baseline_paths, limit_per_file=baseline_limit)
    baseline_counts = get_word_counts(baseline_texts)  # Corpus 2
    filtered_baseline_counts = filter_word_counts(baseline_counts, min_count=5)

    baseline_avg_len = compute_avg_word_length(baseline_texts)
    baseline_fk_score, baseline_fk_grade = compute_corpus_fk_metrics(baseline_texts)

    print("=" * 80)
    limit_note = f" (capped to first {baseline_limit}/dataset)" if baseline_limit else ""
    print(f"BASELINE METRICS: {model_name} | {dataset_name}{limit_note} ({len(baseline_texts)} responses, {sum(baseline_counts.values())} total tokens)")
    print(f"  - Datasets:        {', '.join(target_datasets)}")
    print(f"  - Avg Word Length: {baseline_avg_len:.3f} chars")
    print(f"  - FK Score:        {baseline_fk_score:.2f}" if baseline_fk_score else "  - FK Score:        N/A")
    print(f"  - FK Grade Level:  {baseline_fk_grade if baseline_fk_grade else 'N/A'}")
    print("=" * 80 + "\n")

    if not dialect_files:
        print(f"No dialect files found in {data_dir}")
        return

    summary_rows = []
    log_odds_results = {}

    for dialect_name, file_paths in dialect_files:
        dialect_texts = read_csv_outputs(file_paths)
        if not dialect_texts:
            continue
        dialect_counts = get_word_counts(dialect_texts)  # Corpus 1

        # Text readability metrics
        avg_len = compute_avg_word_length(dialect_texts)
        fk_score, fk_grade = compute_corpus_fk_metrics(dialect_texts)

        # 3. Log-Odds with Dirichlet Prior (filtered to min_count >= 5)
        try:
            filtered_dialect_counts = filter_word_counts(dialect_counts, min_count=5)
            scores = compute_log_odds(filtered_dialect_counts, filtered_baseline_counts)

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
    print("### Summary Comparison Table ###")
    if pd is not None:
        summary_df = pd.DataFrame(summary_rows)
        print(summary_df.to_string(index=False))
    else:
        for r in summary_rows:
            print(r)
    print("\n" + "=" * 80)

    # 5. Print Top Distinctive Words (Log-Odds)
    print("### Distinctive Words (Log-Odds with Dirichlet Prior, min_count >= 5) ###")
    for dialect, res in log_odds_results.items():
        print(f"\n--- {dialect.upper()} vs. BASELINE ---")
        if "error" in res:
            print(f"  Error: {res['error']}")
        else:
            print(f"  Top words characteristic of {dialect}:")
            print(f"    {res['dialect_top']}")
            print("  Top words characteristic of Baseline:")
            print(f"    {res['baseline_top']}")

    # Save metrics with model and dataset in the CSV file name
    output_filename = f"{model_name}_{dataset_name}_metrics.csv"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    if pd is not None:
        summary_df.to_csv(output_path, index=False)
    else:
        import csv
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            w.writeheader()
            w.writerows(summary_rows)
    print(f"\nReadability metrics saved to '{output_path}'.")


if __name__ == "__main__":
    main()
