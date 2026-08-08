import numpy as np

from rag_v2.embedding.bge_embedder import BGEEmbedder, BGEEmbedderConfig, l2_normalize_np


def test_l2_normalize_np_row_wise_and_zero_safe():
    vectors = np.array([[3.0, 4.0], [0.0, 0.0]], dtype="float32")
    normalized = l2_normalize_np(vectors)
    assert np.allclose(normalized[0], np.array([0.6, 0.8], dtype="float32"))
    assert np.allclose(normalized[1], np.array([0.0, 0.0], dtype="float32"))
    assert normalized.dtype == np.float32


def test_prepare_query_adds_instruction_without_loading_model():
    embedder = BGEEmbedder.__new__(BGEEmbedder)
    embedder.config = BGEEmbedderConfig(model_path="dummy", use_query_instruction=True, query_instruction="检索：")
    assert embedder.prepare_query(" 台风预警 ") == "检索：台风预警"


def test_prepare_query_can_disable_instruction_without_loading_model():
    embedder = BGEEmbedder.__new__(BGEEmbedder)
    embedder.config = BGEEmbedderConfig(model_path="dummy", use_query_instruction=False, query_instruction="检索：")
    assert embedder.prepare_query(" 台风预警 ") == "台风预警"
