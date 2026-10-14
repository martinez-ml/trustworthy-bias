import argparse
from concurrent.futures import ProcessPoolExecutor
import os
import sys
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


import stanza
from multivalue import Dialects
from multivalue.BaseDialect import BaseDialect

#10 verified dialects
DIALECTS = {
    "Chicano": Dialects.ChicanoDialect,
    "Colloquial_American": Dialects.ColloquialAmericanDialect,
    "Indian": Dialects.IndianDialect,
    "Appalachian": Dialects.AppalachianDialect,
    "Aboriginal": Dialects.AboriginalDialect,
    "North_of_England": Dialects.NorthEnglandDialect,
    "Ozark": Dialects.OzarkDialect,
    "Southeast_American_Enclave": Dialects.SoutheastAmericanEnclaveDialect,
    "Urban_African_American": Dialects.AfricanAmericanVernacular,
    "Black_South_African": Dialects.BlackSouthAfricanDialect,
}

_ORIG_STANZA_PIPELINE = getattr(stanza, "Pipeline", None)
_ORIG_CREATE_COREF_CLUSTER = BaseDialect.create_coref_cluster

_ORIG_RELATAVIZE_FILE = BaseDialect.relatavize_file


def _smart_relatavize(self, file_name):
    candidate = _ORIG_RELATAVIZE_FILE(self, file_name)
    if os.path.exists(candidate):
        return candidate

    basename = os.path.basename(file_name)
    candidate_root = os.path.join(SCRIPT_DIR, "resources", basename)
    if os.path.exists(candidate_root):
        return candidate_root

    candidate_direct = os.path.join(SCRIPT_DIR, file_name)
    if os.path.exists(candidate_direct):
        return candidate_direct

    return candidate


BaseDialect.relatavize_file = _smart_relatavize


def configure_coref(enable_coref=False):

    if not enable_coref:
        stanza.Pipeline = lambda *args, **kwargs: None
        BaseDialect.create_coref_cluster = lambda self, s: None
    else:
        if _ORIG_STANZA_PIPELINE is not None:
            stanza.Pipeline = _ORIG_STANZA_PIPELINE

        def robust_coref(self, s):
            try:
                return _ORIG_CREATE_COREF_CLUSTER(self, s)
            except Exception:
                return None

        BaseDialect.create_coref_cluster = robust_coref


configure_coref(enable_coref=False)

_WORKER_DIALECT_CACHE = {}


def process_task(task_args):
    """Worker function for an individual (dialect, dataset) pair."""
    dialect_name, filename, df, col, enable_coref = task_args

    try:
        configure_coref(enable_coref)
        BaseDialect.relatavize_file = _smart_relatavize

        if dialect_name not in _WORKER_DIALECT_CACHE:
            _WORKER_DIALECT_CACHE[dialect_name] = DIALECTS[dialect_name]()

        dialect = _WORKER_DIALECT_CACHE[dialect_name]

        output_dir = os.path.join(SCRIPT_DIR, "translations", "cancer-myth", dialect_name)
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, filename)

        df_out = df.copy()
        translated_texts = []
        rules_applied = []

        for idx, text in enumerate(tqdm(df_out[col], desc=f"[{dialect_name}] {filename}")):
            if not isinstance(text, str) or not text.strip():
                translated_texts.append("" if pd.isna(text) else str(text))
                rules_applied.append([])
                continue

            try:
                dialect.executed_rules = {}
                transformed = dialect.transform(text)
                translated_texts.append(transformed)

                rules = getattr(dialect, "executed_rules", {})
                rule_names = [info["type"] for info in rules.values()] if rules else []
                rules_applied.append(rule_names)
            except Exception as e:
                print(f"\n[{dialect_name}][Warning] Row {idx} error: {e}")
                translated_texts.append(text)
                rules_applied.append([])

        df_out[f"original_{col}"] = df[col]
        df_out[f"{col}_{dialect_name}"] = translated_texts
        df_out[col] = translated_texts
        df_out["rules_applied"] = [str(r) for r in rules_applied]
        df_out["has_changes"] = [
            str(orig) != str(trans)
            for orig, trans in zip(df[col], translated_texts)
        ]

        df_out.to_csv(out_path, index=False)
        print(f"[{dialect_name}] Saved: {out_path} ({sum(df_out['has_changes'])}/{len(df_out)} modified)")
        return dialect_name, filename, True

    except Exception as e:
        import traceback
        print(f"\n[ERROR] Task failed for {dialect_name} on {filename}: {e}")
        traceback.print_exc()
        return dialect_name, filename, False


def main():
    default_workers = min(os.cpu_count() or 4, len(DIALECTS))
    parser = argparse.ArgumentParser(description="Multi-Dialect Translation for Cancer-Myth")
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=default_workers,
        help=f"Number of parallel worker processes (default: {default_workers})",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Optional row limit per dataset (useful for quick smoke tests)",
    )
    parser.add_argument(
        "--dialects",
        "-d",
        nargs="*",
        default=None,
        help="Specific dialects to process (default: all 10 dialects)",
    )
    parser.add_argument(
        "--with-coref",
        action="store_true",
        default=False,
        help="Enable Stanza neural coreference (~30s/row; default is disabled for ~6000x speedup)",
    )
    args = parser.parse_args()

    configure_coref(args.with_coref)

    # Determine which dialects to process
    active_dialects = DIALECTS
    if args.dialects:
        valid_dialects = {}
        for d in args.dialects:
            matched = False
            for k, v in DIALECTS.items():
                if d.lower() == k.lower():
                    valid_dialects[k] = v
                    matched = True
                    break
            if not matched:
                print(f"Warning: Unknown dialect '{d}'. Available: {list(DIALECTS.keys())}")
        if valid_dialects:
            active_dialects = valid_dialects

    print("Loading datasets...")
    at_issue_path = os.path.join(SCRIPT_DIR, "datasets", "cancer-myth", "cancer_at_issue_assert.csv")
    not_at_issue_path = os.path.join(SCRIPT_DIR, "datasets", "cancer-myth", "cancer_not_at_issue_assert.csv")

    try:
        hf_df = load_dataset("Cancer-Myth/Cancer-Myth", split="validation").to_pandas()
    except Exception as e:
        print(f"Warning: Could not download Hugging Face Cancer-Myth online ({e}). Attempting offline cache...")
        hf_df = load_dataset("Cancer-Myth/Cancer-Myth", split="validation", download_mode="reuse_cache_if_exists").to_pandas()

    datasets = {
        "cancer_at_issue_assert.csv": {
            "df": pd.read_csv(at_issue_path),
            "target_col": "user_1",
        },
        "cancer_not_at_issue_assert.csv": {
            "df": pd.read_csv(not_at_issue_path),
            "target_col": "user_1",
        },
        "cancer_myth_hf.csv": {
            "df": hf_df,
            "target_col": "question",
        },
    }

    if args.limit and args.limit > 0:
        print(f"Limiting to first {args.limit} rows per dataset for testing...")
        for k in datasets:
            datasets[k]["df"] = datasets[k]["df"].head(args.limit)

    # Parallel tasks (dialects x datasets) for optimal load balancing across CPU cores
    tasks = [
        (dialect_name, filename, item["df"], item["target_col"], args.with_coref)
        for dialect_name in active_dialects.keys()
        for filename, item in datasets.items()
    ]

    print(f"Running translation across {len(active_dialects)} dialects using {args.workers} workers...")
    mode_str = "Enabled (Slow)" if args.with_coref else "Disabled (~6,000x faster, no word mismatch errors)"
    print(f"Stanza neural coreference: {mode_str}\n")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(process_task, tasks))

    successes = sum(1 for r in results if r[2])
    print(f"\nAll tasks finished: {successes}/{len(tasks)} completed successfully.")
    print("Output directory: translations/cancer-myth/")


if __name__ == "__main__":
    main()