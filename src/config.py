# -*- coding: utf-8 -*-
"""모델 이름·경로를 한곳에만 둔다. 다른 파일에서 문자열을 흩뿌리지 않는다."""

from pathlib import Path

WORKS_ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = WORKS_ROOT / "data" / "disaster_manual.csv"
CHROMA_DIR = WORKS_ROOT / "data" / "chroma"
SAMPLE_DIR = WORKS_ROOT / "assets" / "samples"

COLLECTION_NAME = "disaster_manuals"

# CPU only
DEVICE = "cpu"

# Day 1 검색. 기본 BGE-m3, 실패 시 ingest가 Ko-SBERT로 내린다 (ISSUE-02).
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_FALLBACK = "jhgan/ko-sroberta-multitask"

# Day 3 리랭커 (ISSUE-04 확정)
RERANKER_MODEL = "Dongjin-kr/ko-reranker"

# Day 2 Vision. 이 PC에 Ollama가 있으므로 HF transformers 대신 Ollama (ISSUE-03).
VISION_BACKEND = "ollama"
VISION_MODEL = "moondream"

RETRIEVE_K = 15
RERANK_K = 3
# 화면에는 신뢰도 1위만 보여 준다.
# 리랭커 시그모이드 점수가 이보다 낮으면 "신뢰도 낮음" 안내
LOW_SCORE_THRESHOLD = 0.25

VISION_PROMPT = (
    "Analyze this emergency or disaster image.\n"
    "Identify and list the following in Korean, separated by commas:\n"
    "1. Disaster Type (e.g., 침수, 화재, 지진)\n"
    "2. Location/Environment (e.g., 지하주차장, 도로, 실내)\n"
    "3. Immediate Hazard Elements (e.g., 차오르는 물, 짙은 연기, 균열)"
)
