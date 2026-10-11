# -*- coding: utf-8 -*-
"""
이미지 → Vision → 검색 Top 15 → 리랭크 → 신뢰도 1위 원문 카드.

생성형 LLM으로 행동요령을 다시 쓰지 않는다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from PIL import Image

from src.config import RERANK_K, RETRIEVE_K
from src.reranker import load_reranker, rerank
from src.retriever import get_collection, search
from src.template_builder import render, render_vision_panel
from src.vision_parser import parse, tags_to_query


def warmup() -> None:
    """앱 시작 시 1회. 요청마다 모델을 다시 열지 않는다."""
    print("[pipeline] warmup: chroma")
    get_collection()
    print("[pipeline] warmup: retriever embedder")
    search("화재", k=1)
    print("[pipeline] warmup: reranker")
    load_reranker()
    print("[pipeline] warmup 완료 (vision은 첫 이미지에서 Ollama 로드)")


def run(image: Union[str, Path, Image.Image], retrieve_k: int = RETRIEVE_K, rerank_k: int = RERANK_K) -> dict:
    vision = parse(image)
    query = tags_to_query(vision)
    hits = search(query, k=retrieve_k) if query else []
    top = rerank(query, hits, k=rerank_k) if hits else []
    cards_md = render(vision, top)
    return {
        "vision": vision,
        "query": query,
        "hits": top,
        "left_md": render_vision_panel(vision),
        "right_md": cards_md,
    }


def analyze(image: Union[str, Path, Image.Image, None]) -> tuple[str, str]:
    """Gradio 콜백용. 예외는 짧은 한글 메시지."""
    if image is None:
        return "이미지를 올려 주세요.", ""
    try:
        result = run(image)
        return result["left_md"], result["right_md"]
    except Exception as exc:
        import traceback

        traceback.print_exc()
        return f"분석 실패: {exc}", ""
