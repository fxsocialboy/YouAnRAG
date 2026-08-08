"""Centralized path and parameter configuration for RAG V2.

The module intentionally resolves paths from this file location instead of the
current working directory, so scripts and tests can be executed from anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Stage1ChunkParams:
    """Default chunking parameters for stage 1.

    These values are deliberately fixed for the first implementation to keep
    stage 1 convergent and avoid a large parameter-search matrix.
    """

    target_tokens: int = 280
    soft_max_tokens: int = 360
    hard_max_tokens: int = 448
    overlap_tokens: int = 50
    min_tokens: int = 40

    def validate(self) -> None:
        if not (0 <= self.overlap_tokens < self.target_tokens):
            raise ValueError("overlap_tokens must be non-negative and smaller than target_tokens")
        if not (self.min_tokens <= self.target_tokens <= self.soft_max_tokens <= self.hard_max_tokens):
            raise ValueError("token limits must satisfy min <= target <= soft_max <= hard_max")


@dataclass(frozen=True)
class RagV2Config:
    """Resolved filesystem layout for the independent newrag project."""

    project_root: Path
    legacy_rag_dir: Path
    source_markdown_dir: Path
    model_path: Path
    artifacts_dir: Path
    stage1_artifacts_dir: Path
    experiments_dir: Path
    chunk_params: Stage1ChunkParams = Stage1ChunkParams()
    use_query_instruction: bool = True
    query_instruction: str = "为这个句子生成表示以用于检索相关文章："

    @classmethod
    def default(cls) -> "RagV2Config":
        # config.py -> rag_v2 -> src -> newrag
        project_root = Path(__file__).resolve().parents[2]
        legacy_rag_dir = project_root / "legacy_snapshot" / "RAG"
        source_markdown_dir = legacy_rag_dir / "final_mds"
        model_path = Path(
            r"G:\tiaozhanbei\Youan-AI-main\youan-multiagent\multi_agent_server\app\RAG\bge-large-zh-v1.5"
        )
        artifacts_dir = project_root / "artifacts"
        return cls(
            project_root=project_root,
            legacy_rag_dir=legacy_rag_dir,
            source_markdown_dir=source_markdown_dir,
            model_path=model_path,
            artifacts_dir=artifacts_dir,
            stage1_artifacts_dir=artifacts_dir / "stage1",
            experiments_dir=project_root / "experiments",
        )

    def validate(self, require_model: bool = False) -> None:
        self.chunk_params.validate()
        required_dirs = [self.project_root, self.legacy_rag_dir, self.source_markdown_dir, self.experiments_dir]
        if require_model:
            required_dirs.append(self.model_path)
        missing = [str(path) for path in required_dirs if not path.exists()]
        if missing:
            raise FileNotFoundError("missing required path(s): " + "; ".join(missing))

    def ensure_output_dirs(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.stage1_artifacts_dir.mkdir(parents=True, exist_ok=True)


def get_config() -> RagV2Config:
    """Return the default config and ensure stage output directories exist."""

    cfg = RagV2Config.default()
    cfg.ensure_output_dirs()
    return cfg
