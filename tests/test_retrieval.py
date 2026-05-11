import numpy as np

from app.retrieval.faiss_store import FAISSStore


def test_faiss_store_add_search_rebuild(tmp_path) -> None:
    store = FAISSStore(str(tmp_path), embedding_dim=3)
    vectors = np.array([[1.0, 0.0, 0.0], [0.8, 0.2, 0.0]], dtype=np.float32)
    store.add(vectors, [10, 11])
    results = store.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=2)
    assert results
    assert results[0][0] == 10

    store.rebuild(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), [12])
    rebuilt = store.search(np.array([0.0, 1.0, 0.0], dtype=np.float32), top_k=1)
    assert rebuilt[0][0] == 12
