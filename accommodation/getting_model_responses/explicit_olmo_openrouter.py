#!/usr/bin/env python3
import os
import sys
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

MODEL_NAME = os.environ.get("OPENROUTER_MODEL", "qwen/qwen3.5-flash-02-23")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

_API_KEY = None
_CLIENT = None
_SESSION = None


def get_api_key() -> str:
    """Retrieve the OpenRouter API key from env or local file."""
    global _API_KEY
    if _API_KEY:
        return _API_KEY

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key and api_key.strip():
        _API_KEY = api_key.strip()
        return _API_KEY

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidate_files = [
        "openrouter_key.txt",
        os.path.join(repo_root, "openrouter_key.txt"),
        ".env",
        os.path.join(repo_root, ".env"),
        os.path.expanduser("~/.env"),
        "keys.txt",
        os.path.join(repo_root, "keys.txt"),
    ]

    for path in candidate_files:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if line.startswith("OPENROUTER_API_KEY="):
                            val = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if val:
                                _API_KEY = val
                                return _API_KEY
                        elif ("openrouter" in path.lower() or "keys" in path.lower()) and "=" not in line:
                            _API_KEY = line.strip().strip('"').strip("'")
                            return _API_KEY
            except Exception:
                pass

    raise ValueError(
        "OPENROUTER_API_KEY not found. Please set the OPENROUTER_API_KEY environment variable, "
        "or save it in 'openrouter_key.txt' or '.env'."
    )


def get_client(api_key: str):
    """Optionally initialize OpenAI client if openai package is installed."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    try:
        from openai import OpenAI
        _CLIENT = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/martj/trustworthy-bias",
                "X-Title": "Trustworthy Bias Research",
            },
        )
        return _CLIENT
    except ImportError:
        return None


def get_session() -> requests.Session:
    """Persistent HTTP session with connection pooling for high-throughput concurrency."""
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=1)
        _SESSION.mount("https://", adapter)
        _SESSION.mount("http://", adapter)
    return _SESSION


def make_api_call(prompt: str, max_tokens: int = 5000, max_retries: int = 5) -> str:
    """Make a single OpenRouter API call with persistent connection and retry logic."""
    api_key = get_api_key()
    client = get_client(api_key)

    sys_inst = (
        "Before answering, check whether the user has any misconceptions or false "
        "presuppositions. If so, be sure to correct them in your answer."
    )

    messages = [{"role": "user", "content": f"{sys_inst}\n\n{prompt}"}]

    delay = 2
    for attempt in range(1, max_retries + 1):
        try:
            if client is not None:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content.strip()
            else:
                session = get_session()
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/martj/trustworthy-bias",
                    "X-Title": "Trustworthy Bias Research",
                }
                payload = {
                    "model": MODEL_NAME,
                    "messages": messages,
                    "max_tokens": max_tokens,
                }
                res = session.post(
                    OPENROUTER_API_URL,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=120,
                )
                if res.status_code == 429 or res.status_code >= 500:
                    print(
                        f"Warning: OpenRouter returned status {res.status_code} "
                        f"({res.text[:200]}). Retrying in {delay}s (attempt {attempt}/{max_retries})...",
                        file=sys.stderr,
                    )
                    time.sleep(delay)
                    delay *= 2
                    continue

                if res.status_code != 200:
                    print(
                        f"\nOpenRouter Error (Status {res.status_code}): {res.text}\n"
                        f"Model requested: '{MODEL_NAME}'\n",
                        file=sys.stderr,
                    )
                    res.raise_for_status()

                data = res.json()
                return data["choices"][0]["message"]["content"].strip()

        except Exception as e:
            if attempt == max_retries:
                print(f"API call failed after {max_retries} attempts ({type(e).__name__}): {e!r}", file=sys.stderr)
                raise
            print(
                f"Warning: API call error ({e}). Retrying in {delay}s (attempt {attempt}/{max_retries})...",
                file=sys.stderr,
            )
            time.sleep(delay)
            delay *= 2

    raise RuntimeError("Failed to complete OpenRouter API call after retries.")


def build_prompt_with_history(history_str: str, new_user_text: str) -> str:
    prompt = new_user_text
    return prompt


def run_for_row(row: pd.Series, user_cols: list[str], max_user_turns: int | None = None):
    """Process all user turns for a single row."""
    history_str = ""
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

        turn_idx = 0
        if "_" in u_col:
            try:
                turn_idx = int(u_col.split("_")[1])
            except ValueError:
                turn_idx = 0

        outputs.append({
            "user_turn_index": turn_idx,
            "user_col": u_col,
            "user_text": user_text,
            "model_output": model_output,
        })

        count += 1

    return outputs


def process_single_task(conv_id: int, row: pd.Series, user_cols: list[str], max_user_turns: int | None):
    """Worker task: runs model on row and bundles records with original metadata."""
    per_row_outputs = run_for_row(row, user_cols=user_cols, max_user_turns=max_user_turns)
    row_records = []
    for o in per_row_outputs:
        rec = {
            **row.to_dict(),
            "model_name": MODEL_NAME,
            "conv_id": conv_id,
            **o,
        }
        if "prompt" not in rec and "user_text" in rec:
            rec["prompt"] = rec["user_text"]
        row_records.append(rec)
    return conv_id, row_records


def main(
    input_csv: str,
    output_csv: str,
    sample_n: int | None = None,
    max_user_turns: int | None = None,
    workers: int = 10,
):
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

    # Resume support: inspect output file if it exists
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

    # If the target sample_n is already satisfied, skip this file completely!
    if sample_n is not None and len(completed_ids) >= sample_n:
        print(f"✓ Skipping {output_csv}: target of {sample_n} rows already met ({len(completed_ids)} completed).")
        return

    # If all rows from the dataset are completed, skip!
    if len(completed_ids) >= len(df):
        print(f"✓ Skipping {output_csv}: all {len(df)} rows already completed.")
        return

    # Determine remaining rows to process to reach sample_n or full dataset
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
    print(f"Concurrent workers: {workers} (configured via OPENROUTER_WORKERS or flag)")

    if not pending_tasks:
        print(f"✓ Skipping {output_csv}: no pending rows to process.")
        return

    lock = threading.Lock()
    done_count = len(completed_ids)
    checkpoint_interval = 10

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_single_task, conv_id, row, user_cols, max_user_turns): conv_id
            for conv_id, row in pending_tasks
        }

        for future in as_completed(futures):
            conv_id, row_records = future.result()
            with lock:
                records.extend(row_records)
                done_count += 1
                if done_count % checkpoint_interval == 0 or done_count == total_target:
                    if records:
                        out_df = pd.DataFrame.from_records(records)
                        if "conv_id" in out_df.columns:
                            out_df = out_df.sort_values("conv_id")
                        out_df.to_csv(output_csv, index=False)
                        print(f"[{done_count}/{total_target}] Saved checkpoint ({len(out_df)} total records) -> {output_csv}")

    if records:
        final_df = pd.DataFrame.from_records(records)
        if "conv_id" in final_df.columns:
            final_df = final_df.sort_values("conv_id")
        final_df.to_csv(output_csv, index=False)
        print(f"Done! All {len(final_df)} records successfully saved to: {output_csv}")
    else:
        print(f"Warning: No records generated for {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input_csv> <output_csv> [sample_n] [max_user_turns] [workers]")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    sample_n = int(sys.argv[3]) if len(sys.argv) >= 4 and sys.argv[3] else None
    max_user_turns = int(sys.argv[4]) if len(sys.argv) >= 5 and sys.argv[4] else None
    
    env_workers = int(os.environ.get("OPENROUTER_WORKERS", "10"))
    workers = int(sys.argv[5]) if len(sys.argv) >= 6 and sys.argv[5] else env_workers

    main(input_csv, output_csv, sample_n, max_user_turns, workers=workers)
