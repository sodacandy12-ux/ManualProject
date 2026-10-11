# -*- coding: utf-8 -*-
"""
Day 1: 가공 CSV → BGE-m3 임베딩 → Chroma persist.

실행 (works 폴더에서):
    python -m src.ingest
    python -m src.ingest --rebuild
    python -m src.ingest --rebuild --ko-sbert   # BGE-m3가 죽으면 한국어 SBERT
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# works/ 를 기준으로 import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DATA_CSV,
    DEVICE,
    EMBEDDING_FALLBACK,
    EMBEDDING_MODEL,
)


def load_body_rows(csv_path: Path = DATA_CSV) -> pd.DataFrame:
    """본문 행만 읽는다. 소제목(내용 없음)은 인덱싱하지 않는다."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    body = df[df["행구분"] == "본문"].copy()
    body["검색문장"] = body["검색문장"].fillna("").astype(str).str.strip()
    body = body[body["검색문장"] != ""]
    return body.reset_index(drop=True)


def _meta_str(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def rows_to_chroma_payload(df: pd.DataFrame) -> tuple[list[str], list[str], list[dict]]:
    ids, documents, metadatas = [], [], []
    for _, row in df.iterrows():
        doc_id = str(int(row["매뉴얼일련번호"])) if pd.notna(row["매뉴얼일련번호"]) else str(len(ids))
        ids.append(doc_id)
        documents.append(row["검색문장"])
        metadatas.append(
            {
                "doc_id": doc_id,
                "재난상황": _meta_str(row.get("재난상황")),
                "조치단계": _meta_str(row.get("조치단계")),
                "심각도": _meta_str(row.get("심각도")),
                "제목": _meta_str(row.get("제목")),
                "내용": _meta_str(row.get("내용")),
                "적용장소": _meta_str(row.get("적용장소")),
                "섹션제목": _meta_str(row.get("섹션제목")),
                "재난대분류": _meta_str(row.get("재난대분류")),
            }
        )
    return ids, documents, metadatas


def load_embedder(prefer_bge: bool = True):
    """기본은 BGE-m3. 로드 실패 시 한국어 SBERT."""
    from sentence_transformers import SentenceTransformer

    name = EMBEDDING_MODEL if prefer_bge else EMBEDDING_FALLBACK
    print(f"[ingest] 임베딩 로드: {name} ({DEVICE})")
    try:
        model = SentenceTransformer(name, device=DEVICE)
        return model, name
    except Exception as exc:
        if name == EMBEDDING_FALLBACK:
            raise
        print(f"[ingest] BGE-m3 실패 → fallback {EMBEDDING_FALLBACK}: {exc}")
        model = SentenceTransformer(EMBEDDING_FALLBACK, device=DEVICE)
        return model, EMBEDDING_FALLBACK


def build_index(rebuild: bool = False, prefer_bge: bool = True) -> None:
    import chromadb

    df = load_body_rows()
    print(f"[ingest] 본문 행: {len(df)} (기대 424 전후)")
    empty = int((df["검색문장"].str.len() == 0).sum())
    print(f"[ingest] 검색문장 빈 문자열: {empty}건")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        col = client.get_collection(COLLECTION_NAME)
        if col.count() > 0 and not rebuild:
            print(f"[ingest] 이미 {col.count()}건 인덱싱됨. 다시 만들려면 --rebuild")
            return
        client.delete_collection(COLLECTION_NAME)

    embedder, model_name = load_embedder(prefer_bge=prefer_bge)
    ids, documents, metadatas = rows_to_chroma_payload(df)

    print("[ingest] 임베딩 계산 중 (CPU, 수분 걸릴 수 있음)...")
    vectors = embedder.encode(
        documents,
        batch_size=4,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedding_model": model_name},
    )
    # Chroma add 한 번에 너무 크면 나눠 넣는다
    step = 100
    for i in range(0, len(ids), step):
        j = i + step
        collection.add(
            ids=ids[i:j],
            embeddings=vectors[i:j].tolist(),
            documents=documents[i:j],
            metadatas=metadatas[i:j],
        )
        print(f"[ingest] upsert {min(j, len(ids))}/{len(ids)}")

    print(f"[ingest] 완료. 컬렉션 건수={collection.count()}, 모델={model_name}")
    print(f"[ingest] persist: {CHROMA_DIR}")


SAMPLE_QUERIES = [
    "지하 주차장 침수 차량 탈출",
    "실내 화재 연기 대피",
    "산사태 도로 토사 대피",
]


def print_sample_searches(k: int = 5) -> None:
    from src.retriever import search

    for q in SAMPLE_QUERIES:
        print("\n" + "=" * 60)
        print(f"Q: {q}")
        hits = search(q, k=k)
        if not hits:
            print("  (결과 없음)")
            continue
        for i, hit in enumerate(hits, 1):
            print(
                f"  {i}. [{hit['disaster_type']}/{hit['action_stage']}] "
                f"{hit['title']}  (score={hit['score']:.3f})"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--skip-search", action="store_true")
    parser.add_argument(
        "--bge-m3",
        action="store_true",
        help="(기본값) BAAI/bge-m3. 예전 명령과의 호환용.",
    )
    parser.add_argument(
        "--ko-sbert",
        action="store_true",
        help="jhgan/ko-sroberta-multitask 사용. BGE-m3가 충돌할 때.",
    )
    args = parser.parse_args()
    build_index(rebuild=args.rebuild, prefer_bge=not args.ko_sbert)
    if not args.skip_search:
        print_sample_searches()


if __name__ == "__main__":
    main()
