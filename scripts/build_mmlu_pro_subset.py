"""One-off builder for eval/mmlu_pro_subset.jsonl from the public MMLU-Pro benchmark.

Run with: uv run --no-project --with datasets python scripts/build_mmlu_pro_subset.py
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset

SEED = 42
TOTAL = 100
MAX_CHARS = 2500
OUTPUT_PATH = "eval/mmlu_pro_subset.jsonl"
LETTERS = "ABCDEFGHIJ"
INSTRUCTION = (
    "Answer the following multiple-choice question. Finish with a final line in exactly this "
    "form: Answer: X (where X is the letter of the correct option)."
)


def is_usable(row: dict[str, Any]) -> bool:
    """Keep 10-option rows with a consistent answer letter and a bounded length."""
    options = row["options"]
    if len(options) != 10 or any(str(option).strip() == "N/A" for option in options):
        return False
    answer = row["answer"]
    if not (isinstance(answer, str) and len(answer) == 1 and answer in LETTERS):
        return False
    if LETTERS[row["answer_index"]] != answer:
        return False
    return len(row["question"]) + sum(len(option) for option in options) <= MAX_CHARS


def filter_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the usability rules and drop later duplicates of the same question text."""
    seen: set[str] = set()
    kept = []
    for row in rows:
        if not is_usable(row) or row["question"] in seen:
            continue
        seen.add(row["question"])
        kept.append(row)
    return kept


def pick_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pick a per-category quota with a seeded sample, sorted by (category, question_id)."""
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    categories = sorted(by_category)
    base, extra = divmod(TOTAL, len(categories))
    rng = random.Random(SEED)
    picked = []
    for index, category in enumerate(categories):
        quota = base + 1 if index < extra else base
        candidates = sorted(by_category[category], key=lambda row: row["question_id"])
        if len(candidates) < quota:
            raise SystemExit(f"Category {category!r} has {len(candidates)} rows, needs {quota}")
        picked.extend(rng.sample(candidates, quota))
    return sorted(picked, key=lambda row: (row["category"], row["question_id"]))


def build_prompt(row: dict[str, Any]) -> str:
    """Build the full multiple-choice prompt."""
    option_lines = [f"{LETTERS[i]}. {option}" for i, option in enumerate(row["options"])]
    return "\n".join(
        [INSTRUCTION, "", f"Question: {row['question']}", "", "Options:", *option_lines]
    )


def main() -> None:
    """Download, filter, pick, and write the subset; print the counts."""
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    picked = pick_rows(filter_rows([dict(row) for row in dataset]))
    output = Path(OUTPUT_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as file:
        for position, row in enumerate(picked):
            record = {
                "id": f"m{position + 1:03d}",
                "question": build_prompt(row),
                "difficulty": "hard",
                "category": row["category"],
                "split": "tune" if position % 2 == 0 else "test",
                "check": "choice",
                "expected": [row["answer"]],
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    counts: dict[str, int] = defaultdict(int)
    for row in picked:
        counts[row["category"]] += 1
    print(f"wrote {len(picked)} questions to {OUTPUT_PATH}")
    for category in sorted(counts):
        print(f"  {category}: {counts[category]}")


if __name__ == "__main__":
    main()
