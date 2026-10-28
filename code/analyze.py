"""Topic-model analysis of BAI (감사원) audit reports in the housing/urban/land domain.

Pipeline: meta.tsv + data/text/<srno>.txt -> cleaning -> Kiwi noun extraction -> gensim LDA (K chosen by c_v
coherence) -> topic-term tables, doc-topic matrix, period/year topic shares, hot/cold trend test, period
TF-IDF keywords, keyword co-occurrence network (Louvain) -> results/*.csv and figures/*.png
Usage: python3 analyze.py [--kmin 6 --kmax 20 --kstep 2 --seed 42]
"""
import os, re, json, argparse, math, warnings
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')
RES = os.environ.get('BAI_RESULTS', os.path.join(BASE, 'output'))
FIG = os.path.join(RES, 'figures')
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# ---------------------------------------------------------------- settings
PERIODS = [('P1 2003-2012', 2003, 2012), ('P2 2013-2017', 2013, 2017), ('P3 2018-2021', 2018, 2021), ('P4 2022-2026', 2022, 2026)]
MIN_HANGUL = 800           # drop docs with too little Korean text (scanned/empty)
MAX_CHARS = 400000         # cap very long reports (appendices) to keep topics document-level

STOP = set('''감사 감사원 감사결과 보고서 실태 추진 관련 사항 경우 이상 결과 기관 업무 이하 내용 해당 대상 기준 당시 이후 이전 현재 조치 처리 필요 확인 검토 결과 방안 요구 통보 주의 시정 개선 징계 문책 요구사항 처분 처분요구 개별 총괄 분야 개요 목차 별표 목록 페이지 그림 자료 출처 비고 합계 소계 단위 백만원 억원 천원 원 년 월 일 개 건 명 회 차 호 조 항 제 등 및 또는 대한 위한 따른 관한 의한 통한 대해 대하 따라 위하 의하 통하 관하 있음 없음 것임 함 됨 임 위 아래 중 내 외 상 하 전 후 간 별 당 측 측면 부분 정도 수준 가능 실시 수행 운영 관리 추진 시행 실태 점검 운용 현황 계획 방법 범위 배경 목적 실시개요 감사실시 감사대상 감사범위 감사기간 감사인원 감사방법 지적 지적사항 적정 부적정 부당 부적절 미흡 소홀 위반 타당 불합리 부실 과다 과소 누락 지연 미이행 미비 문제 문제점 원인 결과 사례 다음 같이 바와 이번 그간 각각 모두 전체 일부 주요 기타 세부 구체 관계 관계자 담당 담당자 직원 공무원 장 부서 과 국 실 팀 센터 본부 지사 사무소 청 부 처 위원회 위원 회의 결정 의결 승인 보고 보고서 검토서 공문 문서 서류 자료 데이터 정보 시스템 번호 일자 기간 연도 년도 기 차수 단계 과정 절차 방식 체계 제도 규정 법 법령 법률 조문 시행령 시행규칙 지침 기준 요건 조건 사유 이유 근거 취지 의견 입장 판단 평가 분석 조사 확인 산정 산출 계산 적용 반영 고려 참고 제시 제출 작성 보완 수정 변경 추가 삭제 포함 제외 제한 금지 허용 가능 불가 여부 유무 등등 등을 등의 등에 등은 등이 ㄱ ㄴ ㄷ ㅇ 가 나 다 라 마 바 사 아 자'''.split())

ALLOW_SHORT = {'땅', '집', '빚', '삶'}
STOP2 = set('''사실 발생 구성 구분 마련 일반 기본 종합 확보 이용 사용 지정 선정 신청 협의 수립 체결 금액 행정 지방 지역 지급 설치 주식회사 특별 국민 실적 영향 부합 초과 미달 우수 소속 조직 정원 관서 주체 참여 제공 연계 검증 정확 통지서 발급 변경 추가 증가 감소 감액 증액 산정 제출 작성 보고 요청 회신 답변 질의 설명 의견 개최 실시 완료 예정 당초 최종 이후 이전 직후 직전 연간 매년 매월 분기 상반기 하반기 년도 연도 전년 금년 내년 올해 작년 전체 일부 대부분 다수 소수 최대 최소 평균 합계 총계 누계 비율 비중 수준 정도 범위 이상 이하 초과 미만 내외 가량 상당 약간 다소 매우 상대 절대 기존 신규 추가 별도 개별 공통 각종 여러 모든 전반 세부 상세 구체 주요 핵심 중요 일반적 통상 보통 정상 비정상 적정성 타당성 적법 적법성 위법 부당 적정 부적정 미흡 소홀 지연 누락 과다 과소 부실 오류 착오 실수 고의 과실 책임 담당 처리 조치 대책 개선 보완 시정 주의 통보 요구 권고 문책 징계 고발 수사 변상 회수 환수 추징 감액 반환 부과 징수 납부 미납 체납 미징수 미부과 과오납 환급'''.split())


def load_meta():
    m = pd.read_csv(os.path.join(DATA, 'meta.tsv'), sep='\t', dtype=str).fillna('')
    m['year'] = m['openDt'].str[:4].astype(int)
    m['srno'] = m['srno'].astype(str)
    return m


def clean(txt):
    txt = txt[:MAX_CHARS]
    txt = re.sub(r'[ -⁯─-╿＀-￯]', ' ', txt)  # box drawing, fullwidth punctuation
    txt = re.sub(r'[^ㄱ-ㆎ가-힣A-Za-z0-9\s\.\,\(\)·]', ' ', txt)  # keep Hangul, latin, digits
    txt = re.sub(r'\d+', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt)
    return txt


def tokenize(docs):
    from kiwipiepy import Kiwi
    kiwi = Kiwi(num_workers=4)
    for w in ['택지개발', '도시개발', '도시재생', '재개발', '재건축', '정비사업', '공공임대', '임대주택', '공유재산', '국유재산', '개발제한구역',
              '지구단위계획', '도시관리계획', '건축허가', '토지보상', '분양가', '분양보증', '전세', '주거복지', '주거급여', '공시지가', '산업단지',
              '혁신도시', '행정중심복합도시', '한국토지주택공사', '주택도시보증공사', '한국주택금융공사', '국토교통부', '국토해양부', '안전관리', '시설물', '내진']:
        kiwi.add_user_word(w, 'NNP')
    out = []
    for i, d in enumerate(docs):
        toks = []
        for sent in kiwi.tokenize(d, normalize_coda=True):
            if sent.tag in ('NNG', 'NNP', 'SL') and (len(sent.form) >= 2 or sent.form in ALLOW_SHORT):
                f = sent.form
                if f in STOP or re.fullmatch(r'[A-Za-z]{1,2}', f):
                    continue
                toks.append(f)
        out.append(toks)
        if (i + 1) % 25 == 0:
            print(f'  tokenized {i + 1}/{len(docs)}')
    return out


def period_of(y):
    for name, a, b in PERIODS:
        if a <= y <= b:
            return name
    return 'other'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kmin', type=int, default=6)
    ap.add_argument('--kmax', type=int, default=20)
    ap.add_argument('--kstep', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--k', type=int, default=0, help='force K (skip search)')
    ap.add_argument('--retokenize', action='store_true')
    args = ap.parse_args()

    if not os.path.isdir(os.path.join(DATA, 'text')):
        if args.retokenize:
            raise SystemExit('Retokenization requires raw data/text files.')
        df = pd.read_csv(os.path.join(BASE, 'published_results', 'corpus_docs.csv'), dtype={'srno': str})
        df = df.sort_values(['year', 'srno']).reset_index(drop=True)
        print(f'Using published corpus metadata: {len(df)} documents')
    else:
        meta = load_meta()
        rows = []
        for _, r in meta.iterrows():
            p = os.path.join(DATA, 'text', f"{r['srno']}.txt")
            if not os.path.exists(p):
                continue
            txt = open(p, encoding='utf-8', errors='ignore').read()
            hangul = len(re.findall(r'[가-힣]', txt))
            if hangul < MIN_HANGUL:
                continue
            rows.append({'srno': r['srno'], 'year': r['year'], 'title': r['title'], 'audKnd': r['audKnd'], 'field': r['field'], 'chars': len(txt), 'hangul': hangul, 'text': clean(txt)})
        df = pd.DataFrame(rows).sort_values(['year', 'srno']).reset_index(drop=True)
        df['period'] = df['year'].apply(period_of)
        print(f'corpus: {len(df)} docs, years {df.year.min()}-{df.year.max()}')
        print(df.groupby('period').size())
    tok_path = os.path.join(RES, 'tokens.json')
    if os.path.exists(tok_path) and not args.retokenize:
        cache = json.load(open(tok_path, encoding='utf-8'))
        if all(s in cache for s in df['srno']):
            print('loading cached tokens')
            tokens = [cache[s] for s in df['srno']]
        else:
            print('tokenizing...'); tokens = tokenize(df['text'].tolist())
    else:
        print('tokenizing...'); tokens = tokenize(df['text'].tolist())
    tokens = [[w for w in t if w not in STOP2] for t in tokens]
    df['n_tokens'] = [len(t) for t in tokens]
    json.dump({s: t for s, t in zip(df['srno'], tokens)}, open(os.path.join(RES, 'tokens.json'), 'w', encoding='utf-8'), ensure_ascii=False)

    # overall frequency
    cnt = Counter(w for t in tokens for w in t)
    pd.DataFrame(cnt.most_common(300), columns=['term', 'freq']).to_csv(os.path.join(RES, 'term_freq_top300.csv'), index=False, encoding='utf-8-sig')

    # ---------------------------------------------------------------- LDA
    from gensim import corpora
    from gensim.models import LdaModel, CoherenceModel
    dictionary = corpora.Dictionary(tokens)
    dictionary.filter_extremes(no_below=5, no_above=0.7, keep_n=8000)
    corpus = [dictionary.doc2bow(t) for t in tokens]
    print(f'vocab={len(dictionary)}')

    coh_rows = []
    best = None
    ks = [args.k] if args.k else list(range(args.kmin, args.kmax + 1, args.kstep))
    for k in ks:
        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, random_state=args.seed, passes=20, iterations=400, alpha='auto', eta='auto', chunksize=200, eval_every=None)
        cm = CoherenceModel(model=lda, texts=tokens, dictionary=dictionary, coherence='c_v', processes=1)
        cv = cm.get_coherence()
        um = CoherenceModel(model=lda, corpus=corpus, dictionary=dictionary, coherence='u_mass').get_coherence()
        lp = lda.log_perplexity(corpus)
        coh_rows.append({'K': k, 'c_v': cv, 'u_mass': um, 'log_perplexity': lp})
        print(f'K={k} c_v={cv:.4f} u_mass={um:.3f} logppx={lp:.3f}')
        if best is None or cv > best[1]:
            best = (k, cv, lda)
    if not args.k:
        pd.DataFrame(coh_rows).to_csv(os.path.join(RES, 'coherence.csv'), index=False)
    K, cv, lda = best
    print(f'selected K={K} (c_v={cv:.4f})')
    lda.save(os.path.join(RES, 'lda_model'))

    # topic-term table
    tt = []
    for t in range(K):
        terms = lda.show_topic(t, topn=20)
        tt.append({'topic': t + 1, 'terms': ' '.join(w for w, _ in terms), 'weights': ' '.join(f'{p:.4f}' for _, p in terms)})
    pd.DataFrame(tt).to_csv(os.path.join(RES, 'topic_terms.csv'), index=False, encoding='utf-8-sig')

    # doc-topic matrix
    theta = np.zeros((len(corpus), K))
    for i, bow in enumerate(corpus):
        for t, p in lda.get_document_topics(bow, minimum_probability=0):
            theta[i, t] = p
    dt = pd.DataFrame(theta, columns=[f'T{t + 1}' for t in range(K)])
    dt.insert(0, 'period', df['period'])
    dt.insert(0, 'title', df['title'])
    dt.insert(0, 'year', df['year'])
    dt.insert(0, 'srno', df['srno'])
    dt['dominant'] = theta.argmax(1) + 1
    dt['dominant_p'] = theta.max(1)
    dt.to_csv(os.path.join(RES, 'doc_topics.csv'), index=False, encoding='utf-8-sig')

    # representative docs per topic
    rep = []
    for t in range(K):
        idx = np.argsort(-theta[:, t])[:5]
        for i in idx:
            rep.append({'topic': t + 1, 'srno': df.srno[i], 'year': df.year[i], 'title': df.title[i], 'p': round(float(theta[i, t]), 3)})
    pd.DataFrame(rep).to_csv(os.path.join(RES, 'topic_representative_docs.csv'), index=False, encoding='utf-8-sig')

    # period & year shares (mean theta) and dominant counts
    tcols = [f'T{t + 1}' for t in range(K)]
    per = dt.groupby('period')[tcols].mean()
    per.loc['ALL'] = dt[tcols].mean()
    per.to_csv(os.path.join(RES, 'period_topic_share.csv'), encoding='utf-8-sig')
    dom = pd.crosstab(dt['period'], dt['dominant'])
    dom.to_csv(os.path.join(RES, 'period_dominant_counts.csv'), encoding='utf-8-sig')
    yr = dt.groupby('year')[tcols].mean()
    yr.to_csv(os.path.join(RES, 'year_topic_share.csv'), encoding='utf-8-sig')

    # hot/cold: Spearman correlation of yearly mean share with year; plus linear slope
    from scipy.stats import spearmanr, linregress
    hc = []
    for t in range(K):
        s = yr[f'T{t + 1}']
        rho, p = spearmanr(s.index.values, s.values)
        lr = linregress(s.index.values, s.values)
        early = dt[dt.period == PERIODS[0][0]][f'T{t + 1}'].mean()
        late = dt[dt.period == PERIODS[-1][0]][f'T{t + 1}'].mean()
        hc.append({'topic': t + 1, 'mean_share': theta[:, t].mean(), 'spearman_rho': rho, 'p_value': p, 'slope_per_year': lr.slope, 'early_share': early, 'late_share': late,
                   'type': 'hot' if (rho > 0 and p < 0.05) else ('cold' if (rho < 0 and p < 0.05) else 'stable')})
    pd.DataFrame(hc).to_csv(os.path.join(RES, 'topic_trends.csv'), index=False, encoding='utf-8-sig')

    # chi-square: period x dominant topic
    from scipy.stats import chi2_contingency
    chi2, pchi, dof, _ = chi2_contingency(dom.values)
    n = dom.values.sum()
    cramer = math.sqrt(chi2 / (n * (min(dom.shape) - 1)))
    json.dump({'chi2': chi2, 'p': pchi, 'dof': dof, 'n': int(n), 'cramers_v': cramer, 'K': K, 'c_v': cv, 'vocab': len(dictionary), 'n_docs': len(df)}, open(os.path.join(RES, 'summary_stats.json'), 'w'), indent=1)

    # ---------------------------------------------------------------- period TF-IDF (document level)
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(token_pattern=r'\S+', sublinear_tf=True, min_df=5, max_df=0.7)
    X = vec.fit_transform([' '.join(t) for t in tokens])
    vocab = np.array(vec.get_feature_names_out())
    overall = np.asarray(X.mean(axis=0)).ravel()
    tf_rows = []
    for p, _, _ in PERIODS:
        mask = (df.period == p).values
        m = np.asarray(X[mask].mean(axis=0)).ravel()
        top = np.argsort(-m)[:25]
        ratio = (m + 1e-6) / (overall + 1e-6)
        dfp = np.asarray((X[mask] > 0).sum(axis=0)).ravel()
        cand = np.where(dfp >= max(3, 0.1 * mask.sum()))[0]
        dist = cand[np.argsort(-ratio[cand])][:20]
        tf_rows.append({'period': p, 'n_docs': int(mask.sum()), 'top_tfidf': ' '.join(f'{vocab[j]}({m[j]:.3f})' for j in top), 'distinctive': ' '.join(f'{vocab[j]}(x{ratio[j]:.1f})' for j in dist)})
    pd.DataFrame(tf_rows).to_csv(os.path.join(RES, 'period_tfidf_top.csv'), index=False, encoding='utf-8-sig')

    # ---------------------------------------------------------------- co-occurrence network (document-level, association-weighted)
    import networkx as nx
    NTOP = 70
    docfreq = Counter(w for t in tokens for w in set(t))
    top_terms = [w for w, _ in cnt.most_common(NTOP)]
    idx = {w: i for i, w in enumerate(top_terms)}
    co = np.zeros((NTOP, NTOP))
    for t in tokens:
        present = sorted({idx[w] for w in set(t) if w in idx})
        for a in range(len(present)):
            for b in range(a + 1, len(present)):
                co[present[a], present[b]] += 1
    co = co + co.T
    N = len(tokens)
    assoc = np.zeros_like(co)   # Jaccard-like normalized co-occurrence
    for i in range(NTOP):
        for j in range(NTOP):
            if i != j:
                dfi, dfj = docfreq[top_terms[i]], docfreq[top_terms[j]]
                assoc[i, j] = co[i, j] / (dfi + dfj - co[i, j]) if (dfi + dfj - co[i, j]) > 0 else 0
    G = nx.Graph()
    for w in top_terms:
        G.add_node(w, freq=cnt[w], df=docfreq[w])
    KNN = 4
    for i in range(NTOP):
        nbrs = np.argsort(-assoc[i])[:KNN]
        for j in nbrs:
            if assoc[i, j] > 0:
                G.add_edge(top_terms[i], top_terms[j], weight=float(assoc[i, j]), co=float(co[i, j]))
    comms = nx.community.louvain_communities(G, weight='weight', seed=args.seed, resolution=1.0)
    cmap = {w: ci for ci, c in enumerate(sorted(comms, key=len, reverse=True)) for w in c}
    deg = nx.degree_centrality(G)
    btw = nx.betweenness_centrality(G, weight=None)
    try:
        eig = nx.eigenvector_centrality(G, weight='weight', max_iter=5000)
    except Exception:
        eig = {w: float('nan') for w in G.nodes}
    net = pd.DataFrame([{'term': w, 'freq': cnt[w], 'doc_freq': docfreq[w], 'community': cmap.get(w, -1), 'degree': deg[w], 'betweenness': btw[w], 'eigenvector': eig[w]} for w in top_terms]).sort_values(['community', 'degree'], ascending=[True, False])
    net.to_csv(os.path.join(RES, 'network_nodes.csv'), index=False, encoding='utf-8-sig')
    json.dump({'nodes': G.number_of_nodes(), 'edges': G.number_of_edges(), 'density': nx.density(G), 'modularity': nx.community.modularity(G, comms, weight='weight'), 'n_comms': len(comms), 'knn': KNN}, open(os.path.join(RES, 'network_stats.json'), 'w'), indent=1)

    # ---------------------------------------------------------------- figures
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
    plt.rcParams['axes.unicode_minus'] = False

    cdf = pd.read_csv(os.path.join(RES, 'coherence.csv')) if os.path.exists(os.path.join(RES, 'coherence.csv')) else pd.DataFrame(coh_rows)
    if len(cdf) > 1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(cdf.K, cdf.c_v, marker='o')
        ax.axvline(K, ls='--', c='gray')
        ax.set_xlabel('토픽 수 (K)'); ax.set_ylabel('일관성 (c_v)'); ax.set_title('토픽 수별 일관성 점수')
        fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig1_coherence.png'), dpi=200); plt.close(fig)

    labels = [f'T{t + 1}' for t in range(K)]
    fig, ax = plt.subplots(figsize=(8, 4))
    per.drop('ALL').plot(kind='bar', stacked=True, ax=ax, colormap='tab20', width=0.7)
    ax.set_ylabel('평균 토픽 비중'); ax.set_xlabel(''); ax.set_title('시기별 토픽 구성'); ax.legend(ncol=4, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18))
    plt.xticks(rotation=0)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig2_period_topic_share.png'), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    yr3 = yr.rolling(3, center=True, min_periods=1).mean()
    for t in range(K):
        ax.plot(yr3.index, yr3[f'T{t + 1}'], label=f'T{t + 1}', lw=1.5)
    ax.set_xlabel('연도'); ax.set_ylabel('평균 토픽 비중(3년 이동평균)'); ax.set_title('토픽 비중의 연도별 추이')
    ax.legend(ncol=5, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig3_topic_trends.png'), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 9))
    pos = nx.spring_layout(G, k=0.6, seed=args.seed, weight='weight')
    sizes = [300 + 25 * math.sqrt(cnt[w]) for w in G.nodes]
    colors = [cmap.get(w, -1) for w in G.nodes]
    wmax = max(d['weight'] for _, _, d in G.edges(data=True))
    ew = [0.3 + 3 * G[u][v]['weight'] / wmax for u, v in G.edges]
    nx.draw_networkx_edges(G, pos, alpha=0.25, width=ew, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors, cmap='tab10', alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family=plt.rcParams['font.family'], ax=ax)
    ax.set_title('핵심 키워드 동시출현 네트워크 (상위 70개 명사, Louvain 군집)'); ax.axis('off')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig4_network.png'), dpi=200); plt.close(fig)

    # corpus description
    desc = df.groupby('year').agg(n=('srno', 'size'), mean_hangul=('hangul', 'mean')).reset_index()
    desc.to_csv(os.path.join(RES, 'corpus_by_year.csv'), index=False)
    df.drop(columns=['text'], errors='ignore').to_csv(os.path.join(RES, 'corpus_docs.csv'), index=False, encoding='utf-8-sig')
    print('done')


if __name__ == '__main__':
    main()
