# -*- coding: utf-8 -*-
"""Ko-reranker Cross-Encoder. 후보 10~15개 → 신뢰도 1위."""

from __future__ import annotations

import math

from src.config import DEVICE, RERANK_K, RERANKER_MODEL

_model = None


def _sigmoid(x: float) -> float:
    # 오버플로 방지
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def load_reranker():
    global _model
    if _model is not None:
        return _model
    from sentence_transformers import CrossEncoder

    print(f"[reranker] 로드: {RERANKER_MODEL} ({DEVICE})")
    _model = CrossEncoder(RERANKER_MODEL, device=DEVICE)
    return _model


def rerank(query: str, hits: list[dict], k: int = RERANK_K) -> list[dict]:
    """query vs title+content (또는 search_text). score는 0~1 시그모이드."""
    if not hits:
        return []
    model = load_reranker()
    pairs = []
    for hit in hits:
        passage = hit.get("search_text") or (
            f"{hit.get('title', '')}\n{hit.get('content', '')}"
        )
        pairs.append([query, passage])
    raw_scores = model.predict(pairs, batch_size=4, show_progress_bar=False)
    ranked = []
    raw_list = [float(x) for x in raw_scores]
    already_prob = bool(raw_list) and all(0.0 <= x <= 1.0 for x in raw_list)
    for hit, raw in zip(hits, raw_list):
        item = dict(hit)
        retrieve = float(hit.get("score") or 0.0)
        item["retrieve_score"] = retrieve
        item["raw_score"] = raw
        rr = raw if already_prob else _sigmoid(raw)
        item["rerank_score"] = rr
        # 짧은 Vision 태그에 reranker가 0에 가까우면 1차 검색 점수를 카드 %에 쓴다.
        item["score"] = rr if rr >= 0.2 else retrieve
        ranked.append(item)
    ranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:k]


if __name__ == "__main__":
    from src.retriever import search

    q = "실내 화재 연기 대피"
    hits = search(q, k=10)
    top = rerank(q, hits, k=3)
    for i, h in enumerate(top, 1):
        print(f"{i}. {h['score']:.3f}  {h['title']}")
