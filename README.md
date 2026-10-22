# 주택·도시·국토 정책의 집행 쟁점은 어떻게 변해 왔는가
## 감사원 감사보고서(2003~2026)에 대한 토픽모델링 분석 — 재현 패키지

이 저장소는 감사원 감사보고서 가운데 건설/주택 분야 239건(2003~2026)을 대상으로 한 LDA 토픽모델링 분석의 코드와 결과 자료를 담고 있다. 논문 투고용 재현 자료이며, 감사보고서 원문(PDF/HWP)과 추출 본문은 감사원 저작물이므로 포함하지 않는다. 원문은 `data/meta.tsv`의 목록과 아래 절차로 누구나 다시 내려받을 수 있다.

## 구성
| 경로 | 내용 |
|---|---|
| `code/extract_text.py`, `code/hwp5_text.py` | PDF/HWP(배포용 암호화 문서 포함)에서 본문 추출 |
| `code/analyze.py` | Kiwi 형태소 분석 → gensim LDA(K 탐색, c_v/u_mass) → 시기·연도별 토픽 비중, Spearman 추세검정, χ² 교차분석, 문서 수준 TF-IDF, 동시출현 네트워크(Jaccard, kNN=4, Louvain) |
| `code/make_figures.py`, `code/make_figures_bw.py` | 논문 그림(컬러 / 흑백 인쇄용) 생성 |
| `data/meta.tsv` | 수집 대상 358건 메타데이터(일련번호, 공개일, 감사종류, 분야, 제목, 첨부파일) |
| `data/text_stats.json` | 본문 추출 결과 요약(241건) |
| `results/tokens.json.gz` | 문서별 형태소 분석 결과(명사 토큰; 압축 해제 후 사용) |
| `results/` | 일관성 점수, 토픽별 상위 단어, 문서-토픽 분포, 시기·연도별 비중, 추세검정, 대표 문서, 정책 영역(D1~D6) 집계, 감사종류 구성, 네트워크 지표 |
| `figures/`, `figures_color/` | 논문 그림 1~6(흑백 / 컬러) |

## 재현 절차
```bash
pip install -r requirements.txt        # Python 3.13 기준; pdftotext(poppler)도 필요
# 1. 감사원 홈페이지 API로 meta.tsv의 첨부파일을 내려받아 한 폴더에 bai_<srno>_<fileId>.<ext>로 저장
#    목록: GET https://www.bai.go.kr/api/bak/dar/AWUBAKDAR001E?audSphCd=016&audSphDtlCd=017&size=400
#    상세: GET https://www.bai.go.kr/api/bak/dar/AWUBAKDAR001E/{srno}
#    파일: POST https://www.bai.go.kr/api/files/downloadZip  {"fileId": ...}
python3 code/extract_text.py <다운로드 폴더>            # → data/text/<srno>.txt
gunzip -k results/tokens.json.gz                       # 형태소 분석 결과를 그대로 쓰려면
python3 code/analyze.py --kmin 6 --kmax 20 --kstep 2   # K 탐색
python3 code/analyze.py --k 14                         # 본 분석(K=14, seed=42)
python3 code/make_figures.py && python3 code/make_figures_bw.py
```

## 주요 설정
- 분석 대상: 한글 800자 이상 본문 239건(2003~2026), 시기 구분 P1 2003~2012 / P2 2013~2017 / P3 2018~2021 / P4 2022~2026
- 형태소 분석: kiwipiepy 0.24(NNG·NNP·SL), 복합명사 사용자 사전 30개, 불용어 2단계
- LDA: gensim 4.4, K=14, passes=20, iterations=400, alpha/eta auto, random_state=42
- 정책 영역: `results/domain_map.json` (D1 토지이용·개발사업, D2 주택공급·주거복지, D3 주택금융·보증, D4 도시기반시설 건설·운영, D5 건축·시설 안전 및 재해, D6 국토행정·재정집행·건설산업)

## 인용
김승희(2026). 주택·도시·국토 정책의 집행 쟁점은 어떻게 변해 왔는가: 감사원 감사보고서(2003~2026)에 대한 토픽모델링 분석을 중심으로. (투고 중)

## 라이선스
코드는 MIT License, 결과 자료는 CC BY 4.0. 감사보고서 원문의 저작권은 감사원에 있다.
