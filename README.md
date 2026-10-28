# 감사보고서에 나타난 주택·도시 정책의 집행 쟁점과 변화
## 감사원 감사보고서(2003~2026) 토픽모델링·시기별 키워드 네트워크 분석 — 재현 패키지 (v1.1, 2026-10-29)

백효진(한밭대학교 UCRC 연구소)·김승희(강원대학교 부동산학과, 교신저자)

이 저장소는 감사원 감사보고서 가운데 주택·도시·국토 분야 239건(2003~2026)을 대상으로 한 LDA 토픽모델링과 시기별 키워드 네트워크 분석의 코드와 결과 자료를 담고 있다. 감사보고서 원문(PDF/HWP)과 추출 본문은 감사원 저작물이므로 포함하지 않으며, `data/meta.tsv`의 목록과 아래 절차로 누구나 다시 내려받을 수 있다. 원문 없이도 공개된 명사 토큰(`published_results/tokens.json.gz`)만으로 표·검정·네트워크·그림을 전부 재현할 수 있다.

## v1.1에서 달라진 점
- `run_analysis.py` 추가: 공개 토큰으로 표 4-1~4-4, 추세검정, 정책 영역 집계, 감사종류 구성, **시기별(P1~P4) 키워드 동시출현 네트워크·연결중심성·Louvain 군집**, 그림을 한 번에 생성하는 통합 실행기.
- 원본 코드·결과·그림은 `upstream_code/`, `published_results/`, `published_figures*/`에 그대로 보존(v1.0, 커밋 a32fc84). 새 실행 결과는 `output/`에 분리 저장.
- `validation.json`, `validation_manuscript_network.csv`, `lda_refit_validation/`: 원고 수치와 재계산 결과의 대조 기록(연결중심성 16개 값 전부 일치, K=14 재학습 시 문서별 토픽 비중 평균 절대차 1.2e-6).
- 자세한 실행 방법과 해석 범위는 `실행안내.txt`, 대조 결과는 `분석대조_검증결과.txt` 참조.

## 구성
| 경로 | 내용 |
|---|---|
| `run_analysis.py` | 통합 실행기(기본: 공개 토큰 재집계 + 시기별 네트워크; `--refit-lda`, `--search-k` 옵션) |
| `code/analyze.py` | 전처리·Kiwi 명사 추출·LDA·일관성·추세·전체 네트워크(경로·캐시 보완본) |
| `code/extract_text.py`, `code/hwp5_text.py` | PDF/HWP(배포용 암호화 문서 포함) 본문 추출 — 원문 확보 시 사용 |
| `code/make_figures.py`, `code/make_figures_bw.py` | 전체 기간 그림(컬러/흑백) |
| `upstream_code/` | v1.0 원본 코드 |
| `data/meta.tsv`, `data/text_stats.json` | 수집 대상 358건 메타데이터, 본문 추출 요약(241건) |
| `published_results/` | v1.0 결과: 명사 토큰(`tokens.json.gz`), 토픽별 상위 단어, 문서-토픽 분포, 시기·연도별 비중, 추세검정, 정책 영역 집계 등 |
| `published_figures/`, `published_figures_color/` | v1.0 그림(흑백/컬러) |
| `output/` | v1.1 실행 결과: 위 표들의 재집계, 시기별 빈도·TF-IDF, `network_nodes_P*_common.csv`·`network_edges_P*_common.csv`(원고 표 4-4·그림 4-5 방식), `*_period.csv`(시기별 상위어 재선정 방식), `figures/` |
| `lda_refit_validation/` | K=14 재학습 결과와 공개 결과의 대조 |
| `provenance.json` | 기준 커밋, 원본 파일 해시, 보완 범위 |

## 재현 절차
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                       # 검증 환경 버전은 requirements_tested.txt
python run_analysis.py                                # 표·검정·시기별 네트워크·그림 (원문 불필요)
python run_analysis.py --refit-lda                    # LDA(K=14, seed=42) 재학습 포함
python run_analysis.py --search-k --refit-lda         # K=6~20 일관성 탐색 후 K=14
```
원문부터 재구축하려면 `requirements_raw_text.txt`와 poppler(pdftotext)를 설치한 뒤
감사원 API(목록 `GET /api/bak/dar/AWUBAKDAR001E?audSphCd=016&audSphDtlCd=017&size=400`, 상세 `GET /api/bak/dar/AWUBAKDAR001E/{srno}`, 파일 `POST /api/files/downloadZip {"fileId": ...}`)로 `data/meta.tsv`의 첨부파일을 받아 `python code/extract_text.py <폴더>` → `python code/analyze.py --k 14 --retokenize` 순으로 실행한다.

## 주요 설정
- 분석 대상: 한글 800자 이상 본문 239건(2003~2026.9), 시기 P1 2003~2012(39) / P2 2013~2017(75) / P3 2018~2021(69) / P4 2022~2026(56)
- 형태소 분석: kiwipiepy 0.24(NNG·NNP·SL), 복합명사 사용자 사전 30개, 불용어 2단계, 문서빈도 5 미만·70% 초과 어휘 제외(어휘 5,825개, 명사 토큰 1,621,850개)
- LDA: gensim 4.4, K=14, passes=20, iterations=400, alpha/eta auto, random_state=42
- 정책 영역(`published_results/domain_map.json`): D1 토지이용·개발사업, D2 주택공급·주거복지, D3 주택금융·보증, D4 도시기반시설 건설·운영, D5 건축·시설 안전 및 재해, D6 국토행정·재정집행·건설산업
- 시기별 네트워크: 전체 빈도 상위 70개 명사(common), 문서 단위 동시출현 Jaccard, 노드별 상위 4개 이웃의 합집합(무방향), 연결중심성 = 연결 수/69, Louvain(seed=42)

## 해석상 유의점
주제·영역 비중은 선정된 보고서 안에서의 상대적 구성이지 집행 결함의 발생률이 아니다. Spearman p는 다중검정 보정 전 값이다. 감사종류 구성표는 선택편향을 보정한 분석이 아니며, 시기별 키워드 변화가 감사청구 주체나 정책 사건 때문에 생겼음을 검증한 것은 아니다.

## 인용
백효진·김승희(2026). 감사보고서에 나타난 주택·도시 정책의 집행 쟁점과 변화. 『부동산경영』 (투고 중).

## 라이선스
코드는 MIT License(`LICENSE`), 결과 자료는 CC BY 4.0. 감사보고서 원문의 저작권은 감사원에 있다.
