#!/usr/bin/env python3
from openai import AzureOpenAI
import sys
import re
import pandas as pd


AZURE_OPENAI_ENDPOINT =TODO
with open('keys.txt','r') as f:
    AZURE_OPENAI_API_KEY = [line.rstrip('\n') for line in f][0]
DEPLOYMENT_NAME = 'gpt-4o'  # The dep
# Azure OpenAI client setup
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version='2024-02-15-preview'
)
def make_api_call(prompt: str, max_tokens: int = 5000)-> str:
    """Make a single API call with proper error handling."""
    try:

        sys_inst = (
        "Before answering, check whether the user has any misconceptions or false "
        "presuppositions. If so, be sure to correct them in your answer."
    )


        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "user", "content": f"{sys_inst}\n\n{prompt}"}
                ],
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API call failed: {e}")
        return "ERROR"


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
    user_cols =['user_1']
    if sample_n is not None and sample_n < len(df):
        df_sub = df.sample(sample_n, random_state=42).copy()
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

    main(input_csv, output_csv, sample_n)

