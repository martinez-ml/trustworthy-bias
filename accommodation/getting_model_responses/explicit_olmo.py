#!/usr/bin/env python3
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import re
import pandas as pd
import torch

MODEL_NAME = "allenai/Olmo-3-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

def make_api_call(prompt: str, max_tokens: int = 5000)-> str:
    """Make a single API call with proper error handling."""
    try:

        sys_inst = (
        "Before answering, check whether the user has any misconceptions or false "
        "presuppositions. If so, be sure to correct them in your answer."
    )

        messages = [{"role": "user", "content": f"{sys_inst}\n\n{prompt}"}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)
        input_length = inputs["input_ids"].shape[1]
        return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
    except Exception as e:
        print(f"Model generation failed ({type(e).__name__}): {e!r}", file=sys.stderr)
        raise


def build_prompt_with_history(history_str: str, new_user_text: str) -> str:

    prompt = new_user_text
    return prompt


def run_for_row(row: pd.Series, user_cols: list[str], max_user_turns: int | None = None):
    """
    For a single conversation row:
      - Iterate over user_1, user_2, ...
      - At each user turn, call Gemini to produce mental model + reply.
      - Accumulate history so later turns condition on earlier ones.
    """
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
        model_output = make_api_call(prompt, max_tokens=5000)

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

        count += 1

    return outputs


def main(input_csv: str, output_csv: str, sample_n: int | None = None, max_user_turns: int | None = None):
    df = pd.read_csv(input_csv)
    user_cols = ["user_1"]
    candidate_cols = [
        "prompt",
        "original_post",
        "question",
        "flipped_story",
        "sentence",
        "user_1",
        "input",
        "text",
    ]
    for candidate in candidate_cols:
        if candidate in df.columns:
            user_cols = [candidate]
            break
    else:
        ignore_cols = {"", "Unnamed: 0", "Unnamed: 0.1", "id", "conv_id", "rules_applied", "has_changes"}
        for col in df.columns:
            if col not in ignore_cols and df[col].dtype == object:
                user_cols = [col]
                break
    print(f"[{os.path.basename(input_csv)}] Using input column: {user_cols}")
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    records = []
    completed_ids = set()
    if os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            if "conv_id" in existing_df.columns:
                completed_ids = set(existing_df["conv_id"].dropna().astype(int))
                records = existing_df.to_dict(orient="records")
                print(f"Resuming: found {len(completed_ids)} already-completed rows in {output_csv}")
        except Exception:
            records = []
            completed_ids = set()

    if sample_n is not None and len(completed_ids) >= sample_n:
        print(f"✓ Skipping {output_csv}: target of {sample_n} rows already met ({len(completed_ids)} completed).")
        return

    if len(completed_ids) >= len(df):
        print(f"✓ Skipping {output_csv}: all {len(df)} rows already completed.")
        return

    uncompleted_rows = [
        (conv_id, row) for conv_id, row in df.iterrows() if conv_id not in completed_ids
    ]

    if sample_n is not None:
        needed = sample_n - len(completed_ids)
        pending_tasks = uncompleted_rows[:needed]
        total_target = sample_n
    else:
        pending_tasks = uncompleted_rows
        total_target = len(df)

    print(f"Target: {total_target} rows | Already done: {len(completed_ids)} | To process: {len(pending_tasks)}")

    if not pending_tasks:
        print(f"✓ Skipping {output_csv}: no pending rows to process.")
        return

    for conv_id, row in pending_tasks:
        per_row_outputs = run_for_row(
            row,
            user_cols=user_cols,
            max_user_turns=max_user_turns,
        )
        for o in per_row_outputs:
            rec = {
                **row.to_dict(),
                "model_name": MODEL_NAME,
                "conv_id": conv_id,
                **o,
            }
            if "prompt" not in rec and "user_text" in rec:
                rec["prompt"] = rec["user_text"]
            records.append(rec)

        if records:
            out_df = pd.DataFrame.from_records(records)
            if "conv_id" in out_df.columns:
                out_df = out_df.sort_values("conv_id")
            out_df.to_csv(output_csv, index=False)
            print(f"Saved progress after conv_id={conv_id}; total rows so far: {len(out_df)}")

    print(f"Done. Final rows written: {len(records)}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    sample_n = int(sys.argv[3]) if len(sys.argv) >= 4 else None

    main(input_csv, output_csv, sample_n)
