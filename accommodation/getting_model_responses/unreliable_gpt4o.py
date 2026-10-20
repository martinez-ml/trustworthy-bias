
import pandas as pd
import time
import re
import numpy as np
from tqdm import tqdm
from openai import AzureOpenAI
import threading
import pickle
import os
import argparse
import sys
from typing import Tuple, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AZURE_OPENAI_ENDPOINT =TODO 
with open('keys.txt','r') as f:
    AZURE_OPENAI_API_KEY = [line.rstrip('\n') for line in f][0]
DEPLOYMENT_NAME = 'gpt-4o'  # The dep
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version='2024-02-15-preview'
)
def make_api_call(prompt: str, max_token: int = 5000)-> str:
    """Make a single API call with proper error handling."""
    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_token,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"API call failed: {e}")
        return "ERROR"


def build_prompt_with_history(history_str: str, new_user_text: str) -> str:
    prompt = '[Note: The following information is from an unreliable narrator.]\n' + new_user_text
    return prompt



def run_for_row(row: pd.Series, user_cols: list[str], max_user_turns: int | None = None):
    history_str = ""  # plain text "User: ..., AI: ..." history
    outputs = []

    count = 0
    for u_col in user_cols:
        if max_user_turns is not None and count >= max_user_turns:
            break

        if u_col not in row.index:
            continue

        user_text = row[u_col]
        if not isinstance(user_text, str) or not user_text.strip():
            continue

        user_text = user_text.strip()

        prompt = build_prompt_with_history(history_str, user_text)
        model_output = make_api_call(prompt, max_token=5000)
        print(model_output)
        # Record this turn
        turn_idx = 0
        if '_' in u_col:
            turn_idx = int(u_col.split("_")[1])
        outputs.append({
            "user_turn_index": turn_idx,
            "user_col": u_col,
            "user_text": user_text,
            "model_output": model_output,
        })

        # Update history for next turn
        if history_str:
            history_str += "\n"
        history_str += f"User: {user_text}\nAI: {model_output}"

        count += 1

    return outputs


def main(input_csv: str, output_csv: str, sample_n: int | None = None, max_user_turns: int | None = None):
    df = pd.read_csv(input_csv)
    user_cols = ['user_1']
    if 'user_1' not in df.columns:
        user_cols = ['prompt']
    if sample_n is not None and sample_n < len(df):
        df_sub = df.head(sample_n)#sample(sample_n, random_state=42).copy()
    else:
        df_sub = df.copy()

    records = []

    for conv_id, row in df_sub.iterrows():
        per_row_outputs = run_for_row(
            row,
            user_cols=user_cols,
            max_user_turns=max_user_turns,
        )
        for o in per_row_outputs:
            rec = {
                "conv_id": conv_id,
                **o,
            }
            records.append(rec)

        out_df = pd.DataFrame.from_records(records)
        out_df.to_csv(output_csv, index=False)
        print(f"Saved progress after conv_id={conv_id}; total rows so far: {len(out_df)}")

    print(f"Done. Final rows written: {len(records)}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    sample_n = int(sys.argv[3]) if len(sys.argv) >= 4 else None
    max_user_turns = int(sys.argv[4]) if len(sys.argv) >= 5 else None

    main(input_csv, output_csv, sample_n, max_user_turns)

