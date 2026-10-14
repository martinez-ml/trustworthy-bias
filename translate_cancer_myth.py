import argparse
from concurrent.futures import ProcessPoolExecutor
import os
import sys
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import stanza
from multivalue import Dialects
from multivalue.BaseDialect import BaseDialect

# Bypass slow Stanza neural coreference (~30s/it and token mismatch errors)
stanza.Pipeline = lambda *args, **kwargs: None
BaseDialect.create_coref_cluster = lambda self, s: None

# Fallback to root resources if not found in package directory
_orig_rel = BaseDialect.relatavize_file
BaseDialect.relatavize_file = lambda self, fn: _orig_rel(self, fn) if os.path.exists(_orig_rel(self, fn)) else os.path.join("resources", os.path.basename(fn))

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


def translate_dataset(task_args):
    dialect_name, filename, df, col = task_args
    dialect = DIALECTS[dialect_name]()

    out_dir = os.path.join("translations", "cancer-myth", dialect_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    df_out = df.copy()
    translated, rules_applied = [], []

    for text in tqdm(df_out[col], desc=f"[{dialect_name}] {filename}"):
        if isinstance(text, str) and text.strip():
            clean_text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
            try:
                dialect.executed_rules = {}
                transformed = dialect.transform(clean_text)
                translated.append(transformed)
                rules = getattr(dialect, "executed_rules", {})
                rules_applied.append([info["type"] for info in rules.values()] if rules else [])
            except Exception:
                translated.append(text)
                rules_applied.append([])
        else:
            translated.append(text)
            rules_applied.append([])

    # Backup original, update target column, and keep tracking columns
    df_out[f"original_{col}"] = df[col]
    df_out[col] = translated
    df_out["rules_applied"] = [str(r) for r in rules_applied]
    df_out["has_changes"] = [str(orig) != str(trans) for orig, trans in zip(df[col], translated)]

    df_out.to_csv(out_path, index=False)
    print(f"[{dialect_name}] Saved: {out_path} ({sum(df_out['has_changes'])}/{len(df_out)} modified)")


def main():
    parser = argparse.ArgumentParser(description="Multi-Dialect Translation for Cancer-Myth")
    parser.add_argument("--workers", "-w", type=int, default=min(os.cpu_count() or 4, len(DIALECTS)), help="Worker processes")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Limit rows per dataset (for quick testing)")
    args = parser.parse_args()

    print("Loading datasets...")
    datasets = [
        ("cancer_at_issue_assert.csv", pd.read_csv("datasets/cancer-myth/cancer_at_issue_assert.csv"), "user_1"),
        ("cancer_not_at_issue_assert.csv", pd.read_csv("datasets/cancer-myth/cancer_not_at_issue_assert.csv"), "user_1"),
        ("cancer_myth_hf.csv", load_dataset("Cancer-Myth/Cancer-Myth", split="validation").to_pandas(), "question"),
    ]

    if args.limit:
        datasets = [(fn, df.head(args.limit), col) for fn, df, col in datasets]

    tasks = [
        (dialect_name, filename, df, col)
        for dialect_name in DIALECTS
        for filename, df, col in datasets
    ]

    print(f"Translating {len(datasets)} datasets into {len(DIALECTS)} dialects using {args.workers} workers...\n")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(translate_dataset, tasks))

    print("\nAll translations completed! Saved under translations/cancer-myth/")


if __name__ == "__main__":
    main()