"""The training prompts of the two instruments: rows of data/dapo17k_boxed.parquet in the order of metrics/prompts.jsonl,
rendered with the chat template of the model's tokenizer.  The full-support instrument uses all 512 prompts, the four-branch
instrument the 288 prompts from offset 224."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PARQUET = os.path.join(HERE, "..", "data", "dapo17k_boxed.parquet")
LISTING = os.path.join(HERE, "metrics", "prompts.jsonl")


def load_prompts(model, offset=0, count=512):
    """[{'qid', 'prompt_ids', 'gt'}] for the prompts offset .. offset + count - 1 of the listing."""
    import pandas as pd
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    df = pd.read_parquet(PARQUET)
    out = []
    with open(LISTING) as f:
        listing = [json.loads(line) for line in f]
    for entry in listing[offset:offset + count]:
        row = df.iloc[entry["row"]]
        enc = tok.apply_chat_template([dict(m) for m in row["prompt"]], add_generation_prompt=True, tokenize=True)
        ids = list(enc["input_ids"]) if hasattr(enc, "keys") else list(enc)
        out.append({"qid": entry["qid"], "prompt_ids": ids, "gt": str(row["reward_model"]["ground_truth"])})
    return out
