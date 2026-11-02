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

    out_dir = os.path.join("translations", "sage-eval", dialect_name)
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
    df_out["rules_applied"] = [str(r) for r in rules_applied]
    df_out["has_changes"] = [str(orig) != str(trans) for orig, trans in zip(df[col], translated)]

    df_out.to_csv(out_path, index=False)
    print(f"[{dialect_name}] Saved: {out_path} ({sum(df_out['has_changes'])}/{len(df_out)} modified)")


def main():
    parser = argparse.ArgumentParser(description="Multi-Dialect Translation for SAGE-Eval")
    parser.add_argument("--workers", "-w", type=int, default=min(os.cpu_count() or 4, len(DIALECTS)), help="Worker processes")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Limit rows (for quick testing)")
    parser.add_argument("--input-file", "-i", type=str, default=None, help="Path to local SAGE-Eval CSV (optional)")
    args = parser.parse_args()

    if args.input_file and os.path.exists(args.input_file):
        df = pd.read_csv(args.input_file)
    else:
        print("Loading YuehHanChen/SAGE-Eval from Hugging Face...")
        ds = load_dataset("YuehHanChen/SAGE-Eval")
        split = "test" if "test" in ds else list(ds.keys())[0] if hasattr(ds, "keys") else None
        df = ds[split].to_pandas() if split else ds.to_pandas()

    if "augmentation" in df.columns:
        unaugmented_mask = df["augmentation"].isin([False, 0, "False", "false", "f"])
        df = df[unaugmented_mask].copy().reset_index(drop=True)
        print(f"Filtered to unaugmented prompts: {len(df)} rows")

    if args.limit:
        df = df.head(args.limit)

    tasks = [(dialect_name, "sage_eval.csv", df, "prompt") for dialect_name in DIALECTS]

    print(f"Translating SAGE-Eval ({len(df)} rows) into {len(DIALECTS)} dialects using {args.workers} workers...\n")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(translate_dataset, tasks))

    print("\nAll translations completed! Saved under translations/sage-eval/")


if __name__ == "__main__":
    main()
