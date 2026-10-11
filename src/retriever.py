# -*- coding: utf-8 -*-
"""BGE-m3 + ChromaDB 1차 검색. query → Top k dict."""

from __future__ import annotations

from typing import Optional

from src.config import CHROMA_DIR, COLLECTION_NAME, DEVICE, RETRIEVE_K

_embedder = None
_embedder_name = None
_collection = None


def _load_embedder():
    global _embedder, _embedder_name
    if _embedder is not None:
        return _embedder
    import chromadb
    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_collection(COLLECTION_NAME)
    name = (col.metadata or {}).get("embedding_model") or "BAAI/bge-m3"
    print(f"[retriever] 임베딩 로드: {name}")
    _embedder = SentenceTransformer(name, device=DEVICE)
    _embedder_name = name
    return _embedder


def get_collection():
    global _collection
    if _collection is not None:
        return _collection
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    names = [c.name for c in client.list_collections()]
    if COLLECTION_NAME not in names:
        raise RuntimeError(
            f"Chroma 컬렉션 '{COLLECTION_NAME}'이 없습니다. "
            "works 폴더에서 `python -m src.ingest` 를 먼저 실행하세요."
        )
    _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def _hit_from_meta(meta: dict, document: str, distance: float) -> dict:
    # cosine distance → 가까울수록 0. score는 높을수록 관련
    score = 1.0 - float(distance)
    return {
        "doc_id": meta.get("doc_id", ""),
        "title": meta.get("제목", ""),
        "content": meta.get("내용", ""),
        "disaster_type": meta.get("재난상황", ""),
        "section_title": meta.get("섹션제목", ""),
        "action_stage": meta.get("조치단계", ""),
        "severity": meta.get("심각도", ""),
        "place": meta.get("적용장소", ""),
        "search_text": document or "",
        "score": score,
    }


def search(query: str, k: int = RETRIEVE_K) -> list[dict]:
    """query: str → list[Hit] Top k."""
    if not query or not str(query).strip():
        return []
    col = get_collection()
    embedder = _load_embedder()
    vec = embedder.encode(
        [query.strip()],
        normalize_embeddings=True,
    )[0].tolist()
    res = col.query(
        query_embeddings=[vec],
        n_results=min(k, max(col.count(), 1)),
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        hits.append(_hit_from_meta(meta or {}, doc or "", dist))
    return hits


if __name__ == "__main__":
    from src.ingest import print_sample_searches

    print_sample_searches()
