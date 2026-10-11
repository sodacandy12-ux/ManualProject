# -*- coding: utf-8 -*-
"""공식 원문 카드. LLM으로 문장을 다시 쓰지 않는다."""

from __future__ import annotations

import re

from jinja2 import Template

from src.config import LOW_SCORE_THRESHOLD

CARD_TEMPLATE = Template(
    """
🚨 **[현장 긴급 감지]** {{ vision.disaster_type or "미분류" }} / {{ vision.location or "장소 미상" }}
⚠️ **감지된 위험 요소:** {{ vision.hazards or (vision.raw_text or "태그 파싱 실패 — 원문 참고") }}

{% if low_confidence %}
> 관련 매뉴얼 신뢰도가 낮습니다. 아래는 검색된 후보이며, 행동요령을 새로 만들지 않습니다.
{% endif %}
{% if not hit %}
매뉴얼을 찾지 못했습니다.
{% else %}
---
**[공식 행동 지침 - 매칭 신뢰도 {{ hit.score_pct }}%]**
- 재난상황: {{ hit.disaster_type }} · 조치단계: {{ hit.action_stage }} · 심각도: {{ hit.severity }}
- 제목: {{ hit.title }}
{% if hit.section_title %}- 섹션: {{ hit.section_title }}{% endif %}
{% if hit.place %}- 적용장소: {{ hit.place }}{% endif %}

**행동 요령:**

{{ hit.content }}

※ 본 지침은 부산광역시 시도긴급구조표준시스템 재난대비국민행동매뉴얼에 기반하여 제공됩니다.
{% endif %}
""".strip()
)


def format_manual_content(text: str) -> str:
    """원문 글자는 그대로 두고, ○ 항목만 줄바꿈한다."""
    raw = (text or "").strip()
    if not raw or "○" not in raw:
        return raw
    parts = re.split(r"○", raw)
    lines = []
    head = parts[0].strip()
    if head:
        lines.append(head)
    for part in parts[1:]:
        item = part.strip()
        if item:
            lines.append(f"○ {item}")
    return "\n\n".join(lines)


def render(vision: dict, hits: list[dict]) -> str:
    best = None
    max_score = 0.0
    for hit in hits or []:
        score = float(hit.get("score") or 0.0)
        if best is None or score > max_score:
            max_score = score
            best = dict(hit)
    if best is not None:
        best["score_pct"] = int(round(max(0.0, min(1.0, max_score)) * 100))
        best["content"] = format_manual_content(best.get("content") or "")
    low = best is not None and max_score < LOW_SCORE_THRESHOLD
    return CARD_TEMPLATE.render(vision=vision or {}, hit=best, low_confidence=low)


def render_vision_panel(vision: dict) -> str:
    v = vision or {}
    return (
        f"**Vision 태그**\n\n"
        f"- 재난유형: {v.get('disaster_type') or '-'}\n"
        f"- 장소: {v.get('location') or '-'}\n"
        f"- 위험요소: {v.get('hazards') or '-'}\n"
        f"- 추론 시간: {v.get('elapsed_sec', '-')}초\n\n"
        f"<details><summary>모델 원문</summary>\n\n{v.get('raw_text') or ''}\n\n</details>"
    )
