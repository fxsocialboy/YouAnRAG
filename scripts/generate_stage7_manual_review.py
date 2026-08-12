"""Create a stratified manual-review worksheet from final Stage7 results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    stage7 = PROJECT_ROOT / "artifacts" / "stage7"
    parser.add_argument("--labeled", type=Path, default=stage7 / "final_labeled_v2_eval.json")
    parser.add_argument("--random", type=Path, default=stage7 / "final_random_v2_eval.json")
    parser.add_argument("--out", type=Path, default=stage7 / "stage7_manual_review.md")
    parser.add_argument("--count", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["results"]


def select_rows(rows: list[dict[str, Any]], *, count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    buckets = [
        [r for r in rows if r.get("status") != "ok"],
        [r for r in rows if r.get("is_fallback")],
        [r for r in rows if r.get("faithfulness") is not None and float(r["faithfulness"]) < 0.7],
        [r for r in rows if r.get("faithfulness") is not None and float(r["faithfulness"]) >= 0.9],
    ]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for bucket in buckets:
        rng.shuffle(bucket)
        for row in bucket[: max(1, count // 5)]:
            if row["query_id"] not in seen:
                selected.append(row)
                seen.add(row["query_id"])
    remaining = [r for r in rows if r["query_id"] not in seen]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, count - len(selected))])
    return selected[:count]


def main() -> None:
    args = parse_args()
    rows = load_rows(args.labeled) + load_rows(args.random)
    selected = select_rows(rows, count=args.count, seed=args.seed)
    lines = [
        "# Stage 7 人工抽查记录",
        "",
        f"固定抽查种子：`{args.seed}`；抽查数量：`{len(selected)}`。",
        "",
        "填写规则：逐条检查答案是否回应问题、关键事实是否被证据支持、引用是否正确、fallback 是否合理。",
        "",
    ]
    for index, row in enumerate(selected, 1):
        evidence = "\n".join(
            f"- [{item.get('citation_id')}] {item.get('source_file')}：{item.get('content', '')[:300]}"
            for item in row.get("evidence", [])
        ) or "- 无"
        lines += [
            f"## {index}. {row.get('query_id')} — {row.get('query')}",
            "",
            f"- 自动状态：`{row.get('status')}`；fallback：`{row.get('is_fallback')}`；faithfulness：`{row.get('faithfulness')}`",
            "- 人工结论：`待填写（通过 / 部分通过 / 不通过）`",
            "- 问题分类：`待填写（无 / 检索 / 生成幻觉 / 引用 / 降级 / 知识库覆盖）`",
            "- 备注：`待填写`",
            "",
            "### 答案",
            "",
            str(row.get("answer", "")),
            "",
            "### 证据",
            "",
            evidence,
            "",
        ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out} ({len(selected)} rows)")


if __name__ == "__main__":
    main()
