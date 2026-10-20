import argparse
from concurrent.futures import ProcessPoolExecutor
import os
import sys
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import stanza
from multivalue import Dialects
from multivalue.BaseDialect import BaseDialect

stanza.Pipeline = lambda *args, **kwargs: None
BaseDialect.create_coref_cluster = lambda self, s: None

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

    out_dir = os.path.join("translations", "elephant", dialect_name)
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

    df_out[f"original_{col}"] = df[col]
    df_out[col] = translated
    df_out["prompt"] = translated
    df_out["rules_applied"] = [str(r) for r in rules_applied]
    df_out["has_changes"] = [str(orig) != str(trans) for orig, trans in zip(df[col], translated)]

    df_out.to_csv(out_path, index=False)
    print(f"[{dialect_name}] Saved: {out_path} ({sum(df_out['has_changes'])}/{len(df_out)} modified)")


def main():
    parser = argparse.ArgumentParser(description="Multi-Dialect Translation for ELEPHANT Benchmark")
    parser.add_argument("--workers", "-w", type=int, default=min(os.cpu_count() or 4, len(DIALECTS)), help="Worker processes")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Limit rows per dataset (for quick testing)")
    parser.add_argument("--dialects", "-d", nargs="*", default=None, help="Specific dialects to process")
    parser.add_argument("--datasets", nargs="*", default=None, help="Specific datasets to process (e.g. OEQ SS)")
    args = parser.parse_args()

    print("Loading ELEPHANT datasets...")
    dataset_configs = [
        ("AITA-NTA-FLIP.csv", "datasets/elephant/AITA-NTA-FLIP.csv", "flipped_story"),
        ("AITA-NTA-OG.csv", "datasets/elephant/AITA-NTA-OG.csv", "original_post"),
        ("AITA-YTA.csv", "datasets/elephant/AITA-YTA.csv", "prompt"),
        ("OEQ.csv", "datasets/elephant/OEQ.csv", "prompt"),
        ("SS.csv", "datasets/elephant/SS.csv", "sentence"),
    ]

    datasets = []
    for fn, path, col in dataset_configs:
        if args.datasets and not any(tag.lower() in fn.lower() for tag in args.datasets):
            continue
        if os.path.exists(path):
            df = pd.read_csv(path)
            if args.limit:
                df = df.head(args.limit)
            datasets.append((fn, df, col))
            print(f"Loaded {fn}: {len(df)} rows (target column: '{col}')")
        else:
            print(f"Warning: File not found: {path}")

    if not datasets:
        print("No datasets found to process. Exiting.")
        sys.exit(1)

    active_dialects = DIALECTS
    if args.dialects:
        active_dialects = {k: v for k, v in DIALECTS.items() if any(d.lower() == k.lower() for d in args.dialects)}

    tasks = [
        (dialect_name, filename, df, col)
        for dialect_name in active_dialects
        for filename, df, col in datasets
    ]

    print(f"\nTranslating {len(datasets)} datasets into {len(active_dialects)} dialects using {args.workers} workers...\n")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(translate_dataset, tasks))

    print("\nAll translations completed! Saved under translations/elephant/")


if __name__ == "__main__":
    main()

