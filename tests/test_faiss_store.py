import numpy as np
import pytest

from rag_v2.stores.faiss_store import FaissStore


def test_faiss_store_from_vectors_and_search():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype="float32")
    metadata = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    store = FaissStore.from_vectors(vectors, metadata)
    assert store.ntotal == 3
    assert store.dim == 2
    hits = store.search(np.array([[1.0, 0.0]], dtype="float32"), top_k=10)[0]
    assert len(hits) == 3
    assert hits[0].index == 0
    assert hits[0].metadata == {"id": "a"}
    assert hits[0].rank == 1


def test_faiss_store_filters_top_k_to_ntotal():
    vectors = np.array([[1.0, 0.0]], dtype="float32")
    store = FaissStore.from_vectors(vectors, [{"id": "a"}])
    hits = store.search(np.array([[1.0, 0.0]], dtype="float32"), top_k=100)[0]
    assert len(hits) == 1
    assert hits[0].index == 0


def test_faiss_store_rejects_metadata_length_mismatch():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    with pytest.raises(ValueError):
        FaissStore.from_vectors(vectors, [{"id": "only-one"}])


def test_faiss_store_rejects_query_dim_mismatch():
    store = FaissStore.from_vectors(np.array([[1.0, 0.0]], dtype="float32"), [{"id": "a"}])
    with pytest.raises(ValueError):
        store.search(np.array([[1.0, 0.0, 0.0]], dtype="float32"), top_k=1)


def test_faiss_store_save_and_load():
    # Prefer project-controlled temp dir because the Windows sandbox may deny
    # pytest's default tmp path in some sessions.
    from pathlib import Path

    base = Path(__file__).resolve().parents[1] / "artifacts" / "stage1" / "test_tmp_faiss"
    base.mkdir(parents=True, exist_ok=True)
    index_path = base / "test.index"
    try:
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
        metadata = [{"id": "a"}, {"id": "b"}]
        store = FaissStore.from_vectors(vectors, metadata)
        store.save(index_path)
        loaded = FaissStore.load(index_path, metadata)
        assert loaded.ntotal == 2
        assert loaded.dim == 2
    finally:
        if index_path.exists():
            index_path.unlink()
