"""Lightweight, model-free evaluation contracts and metrics for Stage7."""

from rag_v2.evaluation.answer_metrics import (
    citation_mapping_metrics,
    fallback_accuracy,
    latency_summary,
    summarize_answer_rows,
)
from rag_v2.evaluation.deepseek_judge import DeepSeekAnswerJudge
from rag_v2.evaluation.models import (
    ALLOWED_DISASTER_TYPES,
    AnswerJudgeResult,
    AtomicFactJudgment,
    EvaluationQuery,
    Stage74RegressionQuery,
    load_evaluation_queries,
    load_stage74_regression_queries,
    validate_evaluation_dataset,
)
from rag_v2.evaluation.retrieval_metrics import (
    evaluate_ranked_results,
    ndcg_at_k,
    reciprocal_rank_at_k,
    recall_at_k,
    summarize_retrieval_rows,
)

__all__ = [
    "AnswerJudgeResult",
    "ALLOWED_DISASTER_TYPES",
    "AtomicFactJudgment",
    "EvaluationQuery",
    "Stage74RegressionQuery",
    "DeepSeekAnswerJudge",
    "citation_mapping_metrics",
    "evaluate_ranked_results",
    "fallback_accuracy",
    "latency_summary",
    "load_evaluation_queries",
    "load_stage74_regression_queries",
    "ndcg_at_k",
    "reciprocal_rank_at_k",
    "recall_at_k",
    "summarize_answer_rows",
    "summarize_retrieval_rows",
    "validate_evaluation_dataset",
]
