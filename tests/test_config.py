from pathlib import Path

import pytest

from rag_v2.config import RagV2Config, Stage1ChunkParams, get_config


def test_default_config_resolves_from_file_location_not_cwd():
    cfg = RagV2Config.default()
    assert cfg.project_root == Path(__file__).resolve().parents[1]
    assert cfg.legacy_rag_dir == cfg.project_root / "legacy_snapshot" / "RAG"
    assert cfg.source_markdown_dir == cfg.legacy_rag_dir / "final_mds"
    assert cfg.stage1_artifacts_dir == cfg.project_root / "artifacts" / "stage1"
    assert cfg.stage2_artifacts_dir == cfg.project_root / "artifacts" / "stage2"
    assert cfg.stage4_artifacts_dir == cfg.project_root / "artifacts" / "stage4"
    assert cfg.qdrant_path == cfg.project_root / "artifacts" / "qdrant_local"
    assert cfg.registry_db_path == cfg.stage2_artifacts_dir / "document_registry.sqlite3"
    assert cfg.qdrant_mode == "local"
    assert cfg.qdrant_url is None
    assert cfg.qdrant_collection == "youan_rag_stage2"
    assert cfg.experiments_dir == cfg.project_root / "experiments"
    assert cfg.reranker_model_name == "BAAI/bge-reranker-base"
    assert cfg.reranker_model_path is None
    assert cfg.reranker_device == "cpu"
    assert cfg.reranker_batch_size == 16
    assert cfg.reranker_max_length == 512
    assert cfg.enable_reranker is False
    assert cfg.enable_mmr is False
    assert cfg.mmr_lambda == 0.7
    assert cfg.mmr_pre_candidates == 20
    assert cfg.mmr_top_k == 10
    assert cfg.enable_hyde is False
    assert cfg.hyde_mode == "rule"
    assert cfg.deepseek_model == "deepseek-chat"
    assert cfg.deepseek_timeout == 8
    assert cfg.deepseek_max_retries == 1
    assert cfg.hyde_max_branches == 1


def test_default_config_existing_required_paths():
    cfg = RagV2Config.default()
    cfg.validate(require_model=False)
    assert cfg.legacy_rag_dir.exists()
    assert cfg.source_markdown_dir.exists()
    assert (cfg.legacy_rag_dir / "chunk_metadata.json").exists()
    assert (cfg.legacy_rag_dir / "faiss_index.index").exists()


def test_get_config_creates_artifacts_dirs():
    cfg = get_config()
    assert cfg.stage1_artifacts_dir.exists()
    assert cfg.stage1_artifacts_dir.is_dir()
    assert cfg.stage2_artifacts_dir.exists()
    assert cfg.stage2_artifacts_dir.is_dir()
    assert cfg.stage4_artifacts_dir.exists()
    assert cfg.stage4_artifacts_dir.is_dir()
    assert cfg.qdrant_path.exists()
    assert cfg.qdrant_path.is_dir()


def test_chunk_params_are_convergent_defaults():
    params = Stage1ChunkParams()
    params.validate()
    assert params.target_tokens == 280
    assert params.soft_max_tokens == 360
    assert params.hard_max_tokens == 448
    assert params.overlap_tokens == 50
    assert params.min_tokens == 40


@pytest.mark.parametrize(
    "params",
    [
        Stage1ChunkParams(target_tokens=50, overlap_tokens=50),
        Stage1ChunkParams(min_tokens=500, target_tokens=280),
        Stage1ChunkParams(target_tokens=400, soft_max_tokens=360),
        Stage1ChunkParams(soft_max_tokens=500, hard_max_tokens=448),
    ],
)
def test_invalid_chunk_params_raise(params):
    with pytest.raises(ValueError):
        params.validate()
