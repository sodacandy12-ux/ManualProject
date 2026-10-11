# DisasterVision-RAG

재난 현장 사진을 올리면 **부산광역시 재난대비 국민행동매뉴얼의 공식 원문**을 찾아 보여 주는 CPU 로컬 데모

**Python · Gradio · Moondream2 (Ollama) · BGE-m3 · ChromaDB · Ko-reranker · Jinja2**

---

## 프로젝트 소개

DisasterVision-RAG는 재난 현장 사진 한 장으로 상황에 맞는 국민행동요령을 찾아 주는 서비스입니다.

사진에서 재난 유형·장소·위험요소를 읽어 내고(Vision), 매뉴얼 데이터에서 관련 항목을 검색한 뒤(Retrieval), 재순위(Re-ranking)를 거쳐 가장 신뢰도 높은 항목을 보여 줍니다.

### 설계 원칙: 행동요령을 생성하지 않는다

재난 상황에서 LLM이 문장을 다시 쓰면 매뉴얼 내용이 왜곡될 수 있습니다. 그래서 **생성형 LLM은 최종 출력에 쓰지 않았습니다.**

- Vision 모델은 사진을 태그로 바꾸는 데만 사용
- 검색·재순위로 공식 매뉴얼 행을 선택
- 화면에는 CSV `내용` 원문을 그대로 출력 (○ 기준 줄바꿈만 적용)
- GPU 없는 환경에서도 동작하도록 전 구간을 CPU 로컬로 구성

---

## 기술 스택

| 구분 | 기술 |
|---|---|
| Language | Python 3.12 |
| Vision | Moondream2 (Ollama) |
| Embedding | BAAI/bge-m3 (폴백: jhgan/ko-sroberta-multitask) |
| Vector DB | ChromaDB (임베디드) |
| Re-ranker | Dongjin-kr/ko-reranker (Cross-Encoder) |
| Template | Jinja2 |
| UI | Gradio |
| Data | pandas, CSV (부산광역시 재난대비 국민행동매뉴얼) |

---

## 주요 기능

- **이미지 기반 상황 파싱**
  - Moondream2가 재난 유형 / 장소 / 위험요소를 추출
  - 영어로 출력되는 경우가 많아 파서가 한글 태그 3종으로 정규화

- **2단계 검색 (Retrieve → Re-rank)**
  - BGE-m3 + ChromaDB로 후보 Top 15 검색
  - Ko-reranker로 Top 3 재순위, 신뢰도 1위 항목 선정

- **원문 보존 카드 출력**
  - 재난상황·조치단계·심각도·제목·적용장소와 함께 매뉴얼 원문 표시
  - 리랭커 점수가 임계값(0.25) 미만이면 "신뢰도 낮음" 안내

- **매뉴얼 데이터 가공**
  - 원본 CSV(506행)에 재난상황 · 조치단계 · 심각도 라벨을 규칙 기반으로 부여
  - 본문 424건을 검색 대상으로 인덱싱 (소제목 82건 제외)

---

## 서비스 흐름

```text
현장 사진 업로드
    ↓
Moondream2 (Ollama)  →  재난유형 / 장소 / 위험요소
    ↓
검색 쿼리 생성
    ↓
BGE-m3 + ChromaDB  →  Top 15
    ↓
Ko-reranker  →  Top 3
    ↓
Jinja2 카드  →  신뢰도 1위 매뉴얼 원문 출력
```

---

## 프로젝트 구조

```text
DisasterVision-RAG/
├── app.py                      # Gradio UI
├── requirements.txt
├── DisasterVision-RAG.pptx     # 발표 자료
├── build_pptx.py               # 발표 자료 생성 스크립트
│
├── src/
│   ├── config.py               # 모델 ID · 경로 · 임계값
│   ├── ingest.py               # 가공 CSV → ChromaDB 인덱싱
│   ├── retriever.py            # 1차 검색 (Top 15)
│   ├── reranker.py             # Cross-Encoder 재순위 (Top 3)
│   ├── vision_parser.py        # Moondream2 호출 · 태그 파싱
│   ├── template_builder.py     # 원문 카드 렌더링
│   ├── pipeline.py             # 전체 파이프라인 연결
│   └── download_samples.py     # 샘플 이미지 다운로드
│
├── data/
│   ├── build_labeled_csv.py    # 재난상황·조치단계·심각도 라벨링
│   ├── disaster_manual.csv     # 가공 데이터
│   └── csv_label_report.txt    # 라벨링 결과 리포트
│
├── Dataset/                    # 원본 · 라벨링 CSV
└── assets/samples/             # 테스트용 샘플 이미지 5장 (출처: ATTRIBUTION.md)
```

벡터 인덱스(`data/chroma/`)는 저장소에 포함하지 않으며 `ingest`로 생성합니다.

---

## 구현 포인트

- **Vision 출력 정규화** — 모델이 영어 한 줄 또는 반복 문장을 내는 경우가 많아, 영→한 매핑과 반복 제거로 태그를 안정화
- **검색 쿼리 구성** — '미상' 등 의미 없는 태그는 쿼리에서 제외하고, 파싱 실패 시 원문 일부로 폴백
- **모델 로드 최적화** — 앱 시작 시 검색기·리랭커를 1회 워밍업하여 요청마다 재로드하지 않음
- **임베딩 폴백** — BGE-m3 로드 실패 시 한국어 SBERT로 대체
- **설정 일원화** — 모델 ID, 경로, Top-K, 임계값을 `config.py`에서만 관리

---

## 실측 결과 (CPU, 2026-09-07)

| 항목 | 결과 |
|---|---|
| 인덱싱 | 본문 424건, BGE-m3, 약 11분 |
| Vision | 첫 장 모델 로드 약 40초, 이후 장당 약 0.2~1초 |
| 샘플 질의 "실내 화재 연기 대피" | 화재 대피 항목 Top 5 모두 반환 |
| 샘플 질의 "산사태 도로 토사 대피" | 산사태 3건 + 태풍/대설 |
| 이미지 → 매뉴얼 | 실내 화재 · 도로 침수 샘플에서 관련 원문 Top 3 확인 |

### 한계

- 산사태 사진을 모델이 침수로 인식하는 경우가 있어, 해당 케이스는 호우·산악 계열 항목으로 검색됨
- 재난상황·조치단계·심각도는 제목 기반 **규칙 라벨**이므로 오분류 가능성 있음
- 경량 Vision 모델(Moondream2)의 한계로 복합 상황 인식 정확도는 제한적

---

## 로컬 실행 방법

### 1. 저장소 복제

```bash
git clone https://github.com/<사용자명>/DisasterVision-RAG.git
cd DisasterVision-RAG
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. Vision 모델 준비 ([Ollama](https://ollama.com) 설치 필요)

```bash
ollama pull moondream
```

### 4. 샘플 이미지 다운로드 (선택) 및 인덱스 생성

```bash
python -m src.download_samples
python -m src.ingest --rebuild
```

CPU에서 수 분~수십 분이 걸릴 수 있으며, 완료되면 샘플 쿼리 3종의 Top 5 제목이 출력됩니다.

### 5. 앱 실행

```bash
python app.py
```

브라우저에서 `http://127.0.0.1:7860` 접속 → 사진 업로드 → **분석** 클릭

---

## 데이터 출처

- 부산광역시 시도긴급구조표준시스템 재난대비국민행동매뉴얼 (2023-08-28 기준)
- 샘플 이미지: Wikimedia Commons (파일별 라이선스는 `assets/samples/ATTRIBUTION.md` 참고)

---

## 향후 개선 방향

- **Vision 모델 고도화** — 더 큰 VLM 또는 파인튜닝으로 산사태·붕괴 등 오인식 케이스 개선
- **라벨 품질 개선** — 규칙 기반 라벨을 검수 데이터로 보정하고 검색 정확도 정량 평가(Top-3 적중률) 추가
- **다중 결과 제공** — 신뢰도 1위뿐 아니라 연관 매뉴얼(예방·복구 단계) 함께 제시
- **위치 연동** — 대피소·재난 기관 정보와 연계
