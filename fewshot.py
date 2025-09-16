# # fewshot.py
#  1. load few-shot exemplars from JSONL
#  2. render exemplars in your existing "Sentence i: ..." + JSON format
#  3. wrap any zero-shot prompt builder into a few-shot builder
#  4. create a shots JSONL from a silver parquet (use validation split)

from __future__ import annotations
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import math

# Few-shot loading & rendering 

def load_shots(path: str, max_n: int, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load up to max_n exemplars from a JSONL file with lines like:
      {"sentences": [...], "selected_indices": [1-based]}
    Returns a shuffled subset (deterministic given seed).
    """
    exs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                exs.append(json.loads(s))
    rng = random.Random(seed)
    rng.shuffle(exs)
    return exs[:max_n]


def render_example(sentences: List[str], indices_1b: List[int]) -> str:
    """
    Render one exemplar in the already established format:
      Input:  "Sentence i: ..."
      Output: {"selected_sentences":[...]}  (JSON)
    """
    lines = [f"Sentence {i+1}: {s}" for i, s in enumerate(sentences)]
    return (
        "Example\n"
        "Input:\n" + "\n".join(lines) + "\n\n"
        "Output:\n" + json.dumps({"selected_sentences": [int(x) for x in indices_1b]}, ensure_ascii=False) + "\n"
    )


def with_few_shot(base_builder, shots: Optional[List[Dict[str, Any]]] = None, max_k: Optional[int] = None):
    """
    Wrap a zero-shot prompt builder with few-shot exemplars.

    Args:
      base_builder: existing function that builds the final prompt for the current doc,
                    e.g. simple_vanilla_prompt(sentences, max_k=None)
      shots: list of {"sentences":[...], "selected_indices":[1-based]} exemplars
      max_k: optional K hint passed to base_builder if it accepts a 'max_k' kwarg

    Returns:
      A new builder function with signature _builder(sentences, **kwargs).
      It accepts an optional 'max_k' kwarg, forwarded to base_builder when supported
    """
    def _builder(sentences: List[str], **kwargs) -> str:
        exemplars_text = ""
        if shots:
            parts = [render_example(ex["sentences"], ex["selected_indices"]) for ex in shots]
            exemplars_text = (
                "You will see several examples. Follow the same input/output format.\n\n"
                + "\n".join(parts)
                + "\n"
            )

        # Prefer caller's max_k if provided; otherwise use the wrapper's max_k
        cap = kwargs.get("max_k", max_k)

        # Call base builder 
        try:
            if hasattr(base_builder, "__code__") and "max_k" in base_builder.__code__.co_varnames:
                query = base_builder(sentences, max_k=cap)
            else:
                query = base_builder(sentences)
        except TypeError:
            try:
                query = base_builder(sentences, max_k=cap)
            except TypeError:
                query = base_builder(sentences)

        return exemplars_text + query

    return _builder


# build shots JSONL from a silver parquet

def make_shots_from_silver(
    silver_path: str,
    out_path: str,
    k: int = 3,
    max_sents: Optional[int] = 8,
    max_chars_per_sent: Optional[int] = 220,
    seed: int = 42,
) -> None:
    """
    Create a few-shot JSONL file from a SILVER parquet (validation split)
    Each line: {"sentences": [...], "selected_indices": [1-based]}.

    Notes:
    - If max_sents is set, truncate to first N sentences and drop indices beyond N.
    - If max_chars_per_sent is set, clip sentence strings to reduce tokens.
    """

    df = pd.read_parquet(silver_path)

    # Stabilize row order to improve reproducibility across environments
    if "id" in df.columns:
        df = df.sort_values("id").reset_index(drop=True)


    if "split" in df.columns:
        if not any(str(x).lower().startswith("valid") for x in df["split"].unique()):
            print(f"[warn] '{silver_path}' not a validation split. Proceeding anyway.")

    rng = random.Random(seed)
    idxs = list(range(len(df)))
    rng.shuffle(idxs)
    pick = idxs[:min(k, len(idxs))]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    written = 0

    def _to_py_int(x):
        # pyarrow scalar to python
        if hasattr(x, "as_py"):
            x = x.as_py()
        # pandas/NumPy NA guard
        if x is None:
            return None
        if isinstance(x, float):
            try:
                if math.isnan(x):
                    return None
            except Exception:
                pass
        try:
            return int(x)
        except Exception:
            return None

    with open(out_path, "w", encoding="utf-8") as f:
        for i in pick:
            row = df.iloc[i]

            # sentences to python str list
            sents = list(row["sentences"])
            if max_sents is not None:
                sents = sents[:max_sents]
            if max_chars_per_sent is not None:
                sents = [str(s)[:max_chars_per_sent] for s in sents]

            # selected_indices may be list/ndarray/arrow
            raw = row["selected_indices"]
            if hasattr(raw, "to_pylist"):           # arrow list
                sel0 = raw.to_pylist()
            elif isinstance(raw, (list, tuple)):
                sel0 = list(raw)
            else:
                try:
                    sel0 = list(raw)                # last resort
                except Exception:
                    sel0 = []

            # normalize to py ints, drop Nones/out-of-range after truncation
            sel0 = [_to_py_int(x) for x in sel0]
            sel0 = [x for x in sel0 if x is not None and 0 <= x < len(sents)]

            # convert to 1-based & sort
            sel1 = sorted({x + 1 for x in sel0})

            if not sents:
                continue

            # ensure only built-in types go into JSON
            ex = {
                "sentences": [str(s) for s in sents],
                "selected_indices": [int(x) for x in sel1],
            }
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            written += 1

    print(f"[make-shots] seed={seed}, k={k}, max_sents={max_sents}, max_chars={max_chars_per_sent}")
    print(f"[make-shots] Wrote {written} exemplars -> {out_path}")

    # Save a small metadata to help reproducibility
    meta = {
        "shots_path": str(out_path),
        "source_silver": str(silver_path),
        "seed": int(seed),
        "k": int(k),
        "max_sents": None if max_sents is None else int(max_sents),
        "max_chars_per_sent": None if max_chars_per_sent is None else int(max_chars_per_sent),
        "num_written": int(written),
    }
    with open(str(out_path) + ".meta.json", "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2)


__all__ = [
    "load_shots",
    "render_example",
    "with_few_shot",
    "make_shots_from_silver",
]
