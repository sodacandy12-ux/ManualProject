# -*- coding: utf-8 -*-
"""
Moondream2 상황 파싱.

이 PC에 Ollama가 있으므로 transformers HF 대신 Ollama로 로드한다.
모델: `ollama pull moondream`  (Moondream2)
"""

from __future__ import annotations

import io
import re
import time
from pathlib import Path
from typing import Union

from PIL import Image

from src.config import VISION_BACKEND, VISION_MODEL, VISION_PROMPT


def _to_pil(image: Union[str, Path, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def _pil_to_jpeg_bytes(pil: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# moondream이 영어로 내거나 같은 단어를 반복할 때 검색 쿼리용으로 줄인다.
_EN_KO = {
    "fire": "화재",
    "flood": "침수",
    "flooding": "침수",
    "water": "물",
    "smoke": "연기",
    "landslide": "산사태",
    "mudslide": "산사태",
    "earthquake": "지진",
    "collapse": "붕괴",
    "crack": "균열",
    "indoor": "실내",
    "outdoor": "실외",
    "road": "도로",
    "street": "도로",
    "city": "도시",
    "urban": "도시",
    "urban area": "도시",
    "forest": "산림",
    "basement": "지하",
    "underground": "지하",
    "parking": "주차장",
    "vehicle": "차량",
    "vehicles": "차량",
    "unknown": "미상",
    "natural disaster": "자연재난",
    "south america": "산지",
    "north america": "산지",
}

_LABEL_PREFIX = re.compile(
    r"^(?:\d+[\.\)\:]\s*)?(?:"
    r"disaster\s*type|location(?:\s*/\s*environment)?|environment|"
    r"immediate\s*hazard\s*elements?|"
    r"재난\s*유형|재난유형|장소|위험\s*요소|위험요소"
    r")\s*[:：\-]\s*",
    re.I,
)


def _collapse_repeats(text: str) -> str:
    """'flood, flood, flood' → 'flood'."""
    prev = None
    out = text
    for _ in range(4):
        out = re.sub(r"\b(\S+)(?:\s*,\s*\1)+\b", r"\1", out, flags=re.I)
        if out == prev:
            break
        prev = out
    return out


def _to_korean_phrase(piece: str) -> str:
    s = _LABEL_PREFIX.sub("", (piece or "").strip().strip("-• "))
    s = re.sub(r"\s+", " ", s).strip(" .,;/")
    if not s:
        return ""
    mapped = _EN_KO.get(s.lower())
    if mapped:
        return mapped
    # 구 안의 영어 단어를 치환 (Fire smoke → 화재 연기)
    def repl(m: re.Match) -> str:
        return _EN_KO.get(m.group(0).lower(), m.group(0))

    return re.sub(r"[A-Za-z]+", repl, s).strip()


def _first_field(text: str, patterns: tuple[str, ...]) -> str:
    for pat in patterns:
        m = re.search(pat, text, flags=re.I | re.S)
        if m:
            val = m.group(1).split("\n")[0]
            val = re.split(
                r",\s*(?:disaster|location|environment|immediate|재난|장소|위험)",
                val,
                maxsplit=1,
                flags=re.I,
            )[0]
            return _to_korean_phrase(_collapse_repeats(val))
    return ""


def parse_tag_text(raw_text: str) -> tuple[str, str, str]:
    """콤마/번호/라벨 목록에서 재난유형, 장소, 위험요소 3개를 뽑는다."""
    text = _collapse_repeats((raw_text or "").strip())
    if not text:
        return "", "", ""

    disaster = _first_field(
        text,
        (r"재난\s*유형\s*[:：]\s*(.+)", r"disaster\s*type\s*[:：]\s*(.+)"),
    )
    location = _first_field(
        text,
        (r"장소\s*[:：]\s*(.+)", r"location(?:\s*/\s*environment)?\s*[:：]\s*(.+)"),
    )
    hazards = _first_field(
        text,
        (
            r"위험\s*요소\s*[:：]\s*(.+)",
            r"immediate\s*hazard\s*elements?\s*[:：]\s*(.+)",
            r"environment\s*[:：]\s*(.+)",
        ),
    )
    if disaster or location or hazards:
        # "Disaster Type: 침수, 화재, 지하주차장"처럼 한 필드에 몰린 경우
        if "," in disaster and (not location or not hazards):
            bits = [b.strip() for b in disaster.split(",") if b.strip()]
            if len(bits) >= 2:
                disaster = bits[0]
                if not location:
                    location = bits[1]
                if not hazards and len(bits) >= 3:
                    hazards = ",".join(bits[2:5])
        disaster = re.sub(r"^\d+[\.\)]\s*", "", disaster).strip()
        return disaster, location, (hazards or "")[:80]

    numbered = re.findall(
        r"(?:^|\n)\s*\d+[\.\)\:]\s*(.+?)(?=(?:\n\s*\d+[\.\)\:])|$)",
        text,
        flags=re.S,
    )
    if len(numbered) >= 3:
        parts = [_to_korean_phrase(p.split("\n")[0]) for p in numbered[:3]]
        return parts[0], parts[1], parts[2][:80]

    compact = re.sub(r"\s+", " ", text)
    chunks = [_to_korean_phrase(c) for c in compact.split(",") if c.strip()]
    chunks = [c for c in chunks if c]
    # 같은 단어 연속 제거
    deduped = []
    for c in chunks:
        if not deduped or deduped[-1] != c:
            deduped.append(c)
    if len(deduped) >= 3:
        return deduped[0], deduped[1], ",".join(deduped[2:5])
    if len(deduped) == 2:
        return deduped[0], deduped[1], ""
    if len(deduped) == 1:
        return deduped[0], "", ""
    return "", "", _to_korean_phrase(text)[:80]


def _ensure_ollama_model() -> None:
    import ollama

    names = []
    try:
        listed = ollama.list()
        models = listed.get("models") or []
        for m in models:
            names.append(m.get("name") or m.get("model") or "")
    except Exception:
        pass
    if any(VISION_MODEL in n for n in names):
        return
    print(f"[vision] Ollama에 {VISION_MODEL} 없음 → pull")
    ollama.pull(VISION_MODEL)


def parse(image: Union[str, Path, Image.Image]) -> dict:
    """이미지 → {disaster_type, location, hazards, raw_text, elapsed_sec}."""
    pil = _to_pil(image)
    t0 = time.perf_counter()
    raw_text = ""

    if VISION_BACKEND != "ollama":
        raise RuntimeError("현재 구현은 VISION_BACKEND=ollama 만 지원한다.")

    import ollama

    _ensure_ollama_model()
    # 너무 큰 사진은 루프·지연을 키우므로 긴 변을 줄인다.
    work = pil.copy()
    work.thumbnail((768, 768))
    jpeg = _pil_to_jpeg_bytes(work)
    resp = ollama.chat(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": VISION_PROMPT,
                "images": [jpeg],
            }
        ],
        options={"temperature": 0, "num_predict": 96},
    )
    if isinstance(resp, dict):
        raw_text = ((resp.get("message") or {}).get("content")) or ""
    else:
        msg = getattr(resp, "message", None)
        raw_text = getattr(msg, "content", None) or ""

    elapsed = time.perf_counter() - t0
    disaster_type, location, hazards = parse_tag_text(raw_text)
    return {
        "disaster_type": disaster_type,
        "location": location,
        "hazards": hazards,
        "raw_text": _collapse_repeats(raw_text.strip())[:500],
        "elapsed_sec": round(elapsed, 2),
    }


def tags_to_query(vision: dict) -> str:
    """검색 쿼리. 태그 파싱 실패 시 raw_text 일부. '미상'은 넣지 않는다."""
    skip = {"", "미상", "unknown", "-", "n/a", "none"}
    parts = []
    seen: set[str] = set()
    for key in ("disaster_type", "location", "hazards"):
        val = (vision.get(key) or "").strip()
        for token in re.split(r"[,/]", val):
            t = token.strip()
            if not t or t.lower() in skip or t in skip:
                continue
            if t in seen:
                continue
            seen.add(t)
            parts.append(t)
    query = " ".join(parts).strip()
    if query:
        return query
    fallback = _collapse_repeats((vision.get("raw_text") or "").strip())
    return fallback[:200]


if __name__ == "__main__":
    import sys

    from src.config import SAMPLE_DIR

    paths = list(SAMPLE_DIR.glob("*.jpg")) + list(SAMPLE_DIR.glob("*.png"))
    if len(sys.argv) > 1:
        paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print("샘플 이미지가 없습니다. assets/samples/ 를 채우세요.")
        raise SystemExit(1)
    for p in paths:
        print("=" * 60)
        print(p.name)
        result = parse(p)
        print(result)
