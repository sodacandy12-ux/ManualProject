# -*- coding: utf-8 -*-
"""
DisasterVision-RAG Gradio 앱.

좌: 사진 업로드 / Vision 태그
우: 매칭 신뢰도 1위 공식 매뉴얼 원문 카드

실행 (works 폴더에서):
    python app.py
"""

import gradio as gr

from src.pipeline import analyze, warmup


def main() -> None:
    print("[app] 모델·인덱스를 미리 로드합니다...")
    warmup()

    with gr.Blocks(title="DisasterVision-RAG") as demo:
        gr.Markdown(
            """
# DisasterVision-RAG
현장 사진 → Moondream2 태그 → BGE-m3 검색 → Ko-reranker → 리랭크 신뢰도 1위 항목 매뉴얼 원문

행동요령은 LLM이 새로 쓰지 않습니다.
            """.strip()
        )
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="현장 사진")
                btn = gr.Button("분석", variant="primary")
                left = gr.Markdown(label="Vision")
            with gr.Column():
                right = gr.Markdown(label="공식 행동 지침")

        btn.click(fn=analyze, inputs=image, outputs=[left, right])

    demo.launch(show_error=True, server_port=7860)


if __name__ == "__main__":
    main()
