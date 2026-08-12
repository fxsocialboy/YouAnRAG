"""Generate the final Stage7 Markdown report from three resumable JSON runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_v2.evaluation.answer_metrics import summarize_answer_rows
from rag_v2.evaluation.retrieval_metrics import summarize_retrieval_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final Stage7 evaluation report")
    stage7 = PROJECT_ROOT / "artifacts" / "stage7"
    parser.add_argument("--legacy-labeled", type=Path, default=stage7 / "final_labeled_legacy_eval.json")
    parser.add_argument("--v2-labeled", type=Path, default=stage7 / "final_labeled_v2_eval.json")
    parser.add_argument("--v2-random", type=Path, default=stage7 / "final_random_v2_eval.json")
    parser.add_argument("--out", type=Path, default=stage7 / "final_report.md")
    parser.add_argument("--summary-out", type=Path, default=stage7 / "final_metrics.json")
    parser.add_argument("--before-metrics", type=Path, default=None)
    return parser.parse_args(argv)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def retrieval_summary(payload: dict[str, Any]) -> dict[str, Any]:
    rows = [row["retrieval_metrics"] for row in payload.get("results", []) if row.get("retrieval_metrics")]
    return summarize_retrieval_rows(rows)


def grouped_answer_summary(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    values = sorted({str(row.get(key, "unknown")) for row in rows})
    return {value: summarize_answer_rows([row for row in rows if str(row.get(key, "unknown")) == value]) for value in values}


def decision_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected_ood = [row for row in rows if row.get("expected_fallback") is True]
    expected_in = [row for row in rows if row.get("expected_fallback") is False or (
        row.get("expected_fallback") is None and row.get("disaster_type") != "out_of_domain"
    )]
    tp = sum(bool(row.get("is_fallback")) for row in expected_ood)
    fn = len(expected_ood) - tp
    fp = sum(bool(row.get("is_fallback")) for row in expected_in)
    tn = len(expected_in) - fp
    reasons: dict[str, int] = {}
    for row in rows:
        if row.get("is_fallback"):
            reason = str(row.get("fallback_reason") or "missing_reason")
            reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "confusion_matrix": {"ood_fallback": tp, "ood_answered": fn, "in_domain_fallback": fp, "in_domain_answered": tn},
        "ood_false_accept_rate": round(fn / len(expected_ood), 4) if expected_ood else None,
        "in_domain_false_rejection_rate": round(fp / len(expected_in), 4) if expected_in else None,
        "fallback_reason_distribution": dict(sorted(reasons.items())),
    }


def _rate(rows: list[dict[str, Any]], predicate) -> float:
    return round(sum(bool(predicate(row)) for row in rows) / len(rows), 4) if rows else 0.0


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _metric_table(legacy: dict[str, Any], v2: dict[str, Any]) -> str:
    metrics = [
        "doc_recall@5", "doc_recall@10", "chunk_recall@5", "chunk_recall@10",
        "doc_mrr@10", "chunk_mrr@10", "doc_ndcg@10", "chunk_ndcg@10",
        "avg_latency_ms", "p50_latency_ms", "p95_latency_ms",
    ]
    lines = ["| 指标 | Legacy | V2 Full | 差值(V2-Legacy) |", "|---|---:|---:|---:|"]
    for key in metrics:
        left, right = legacy.get(key), v2.get(key)
        delta = round(float(right) - float(left), 4) if left is not None and right is not None else None
        lines.append(f"| {key} | {_fmt(left)} | {_fmt(right)} | {_fmt(delta)} |")
    return "\n".join(lines)


def _answer_table(summary: dict[str, Any]) -> str:
    keys = [
        "answer_success_rate", "fallback_rate", "fallback_accuracy", "faithfulness",
        "answer_relevancy", "citation_correctness", "citation_completeness",
        "all_citations_mapped_ratio", "avg_latency_ms", "p50_latency_ms", "p95_latency_ms",
    ]
    return "\n".join(["| 指标 | 结果 |", "|---|---:|"] + [f"| {key} | {_fmt(summary.get(key))} |" for key in keys])


def _examples(rows: list[dict[str, Any]], count: int = 5) -> list[dict[str, Any]]:
    def score(row: dict[str, Any]) -> float:
        if row.get("status") != "ok":
            return -2.0
        if row.get("is_fallback"):
            return -1.0
        value = row.get("faithfulness")
        return float(value) if value is not None else 2.0

    selected = sorted(rows, key=score)[:count]
    return [
        {
            "query_id": row.get("query_id"),
            "query": row.get("query"),
            "status": row.get("status"),
            "fallback": row.get("is_fallback"),
            "faithfulness": row.get("faithfulness"),
            "error": row.get("error") or row.get("judge_error"),
        }
        for row in selected
    ]


def build_report(
    legacy: dict[str, Any], labeled: dict[str, Any], random: dict[str, Any], before_metrics: dict[str, Any] | None = None
) -> tuple[dict[str, Any], str]:
    legacy_retrieval = retrieval_summary(legacy)
    labeled_retrieval = retrieval_summary(labeled)
    labeled_rows = labeled.get("results", [])
    random_rows = random.get("results", [])
    labeled_answer = summarize_answer_rows(labeled_rows)
    random_answer = summarize_answer_rows(random_rows)

    deterministic = {
        "labeled_unknown_citation_count": sum(len(row.get("unknown_citation_ids", [])) for row in labeled_rows),
        "random_unknown_citation_count": sum(len(row.get("unknown_citation_ids", [])) for row in random_rows),
        "labeled_deepseek_actual_rate": _rate(labeled_rows, lambda row: row.get("composer_mode") == "deepseek"),
        "random_deepseek_actual_rate": _rate(random_rows, lambda row: row.get("composer_mode") == "deepseek"),
        "random_in_domain_answer_rate": _rate(
            [r for r in random_rows if r.get("disaster_type") != "out_of_domain"], lambda row: not row.get("is_fallback")
        ),
        "random_ood_fallback_accuracy": _rate(
            [r for r in random_rows if r.get("disaster_type") == "out_of_domain"], lambda row: row.get("is_fallback")
        ),
    }
    metrics = {
        "stage": "7.4",
        "legacy_labeled_retrieval": legacy_retrieval,
        "v2_labeled_retrieval": labeled_retrieval,
        "v2_labeled_answer": labeled_answer,
        "v2_random_answer": random_answer,
        "random_by_query_type": grouped_answer_summary(random_rows, "query_type"),
        "random_by_disaster_type": grouped_answer_summary(random_rows, "disaster_type"),
        "deterministic": deterministic,
        "decision_metrics": decision_metrics(labeled_rows + random_rows),
        "before_fix": before_metrics,
        "low_or_failed_examples": _examples(labeled_rows + random_rows),
    }

    lines = [
        "# YouAn RAG V2 阶段七最终评测报告",
        "",
        "> 本报告由冻结评测结果自动生成。随机鲁棒性集没有人工相关性标签，不报告 Recall。",
        "",
        "## 1. 评测范围",
        "",
        f"- 人工标注集：{len(labeled_rows)} 条；Legacy 与 V2 Full 比较检索指标。",
        f"- 固定种子随机集：{len(random_rows)} 条；评测真实回答、降级、引用、Judge 质量和延迟。",
        "- V2 Full：BGE + Qdrant + BM25/RRF + Cross-Encoder + MMR + Query Rewrite/Multi-Query/条件 HyDE + DeepSeek。",
        "- DeepSeek 同时作为 Composer 和 Judge，指标存在同模型偏差；引用映射和 fallback 指标为确定性指标。",
        "",
        "## 2. 标注集 Legacy vs V2 Full 检索指标",
        "",
        _metric_table(legacy_retrieval, labeled_retrieval),
        "",
        "## 3. 标注集 V2 答案质量",
        "",
        _answer_table(labeled_answer),
        "",
        "## 4. 随机鲁棒性集",
        "",
        _answer_table(random_answer),
        "",
        "确定性补充指标：",
        "",
        f"- 未知引用数量：{deterministic['random_unknown_citation_count']}；",
        f"- 实际使用 DeepSeek Composer 比例：{_fmt(deterministic['random_deepseek_actual_rate'])}；",
        f"- in-domain answer rate：{_fmt(deterministic['random_in_domain_answer_rate'])}；",
        f"- OOD fallback accuracy：{_fmt(deterministic['random_ood_fallback_accuracy'])}。",
        "",
        "## 5. 按灾种拆分",
        "",
        "| 灾种 | 数量 | Faithfulness | Relevancy | Fallback rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for group, values in metrics["random_by_disaster_type"].items():
        lines.append(
            f"| {group} | {values['query_count']} | {_fmt(values.get('faithfulness'))} | "
            f"{_fmt(values.get('answer_relevancy'))} | {_fmt(values.get('fallback_rate'))} |"
        )
    lines += [
        "",
        "## 6. 低分、降级或失败样例",
        "",
        "```json",
        json.dumps(metrics["low_or_failed_examples"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 7. OOD与Fallback决策",
        "",
        "```json",
        json.dumps(metrics["decision_metrics"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 8. Before/After对比",
        "",
        ("已加载before-fix指标快照，详见final_metrics.json中的before_fix字段。" if before_metrics else "未提供before-fix指标快照。"),
        "",
        "## 9. 结果解释边界",
        "",
        "- 知识库仅包含 51 份灾害相关 Markdown，覆盖范围之外的问题应合理降级。",
        "- LLM-as-Judge 分数用于规模化趋势分析，不等同于人工专家审查。",
        "- 最终简历只引用本报告中的真实指标，不把随机集包装成有人工标签的准确率测试。",
        "- 人工抽查记录应另填 `stage7_manual_review.md`，至少抽查总样本的 10%。",
        "",
    ]
    return metrics, "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    before = load(args.before_metrics) if args.before_metrics and args.before_metrics.exists() else None
    metrics, markdown = build_report(load(args.legacy_labeled), load(args.v2_labeled), load(args.v2_random), before)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown, encoding="utf-8")
    args.summary_out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(args.out), "metrics": str(args.summary_out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
