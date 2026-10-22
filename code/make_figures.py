"""Final figures with Korean topic labels (reads results/*.csv)."""
import os, json, math
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import networkx as nx

BASE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(BASE, 'results'), os.path.join(BASE, 'figures')
fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(fp)
plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
plt.rcParams['axes.unicode_minus'] = False

LABELS = {1: '민관합동·산단 개발', 2: '토지·주택 인허가', 3: '시설안전·품질', 4: '하수도 민간투자', 5: '4대강·국책공사', 6: '지역개발 보조사업', 7: '침수·재해예방',
          8: '건축허가·국토정보', 9: '도로·교량 시공', 10: '공공시설 투자심사', 11: '국토부 기관운영', 12: '주택금융·보증', 13: '건설산업·하도급', 14: '공공임대·주거복지'}
json.dump(LABELS, open(os.path.join(RES, 'topic_labels.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

coh = pd.read_csv(os.path.join(RES, 'coherence.csv'))
per = pd.read_csv(os.path.join(RES, 'period_topic_share.csv'), index_col=0).drop('ALL')
yr = pd.read_csv(os.path.join(RES, 'year_topic_share.csv'), index_col=0)
tr = pd.read_csv(os.path.join(RES, 'topic_trends.csv'))
K = len(LABELS)
tcols = [f'T{t}' for t in range(1, K + 1)]
names = [f'T{t} {LABELS[t]}' for t in range(1, K + 1)]

# Fig 1 coherence
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(coh.K, coh.c_v, marker='o', label='c_v')
ax.axvline(14, ls='--', c='gray', lw=1)
ax.set_xlabel('토픽 수 (K)'); ax.set_ylabel('일관성 c_v'); ax.set_title('토픽 수별 일관성 점수')
ax2 = ax.twinx(); ax2.plot(coh.K, coh.u_mass, marker='s', c='tab:red', alpha=0.6, label='u_mass'); ax2.set_ylabel('u_mass', color='tab:red')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig1_coherence.png'), dpi=220); plt.close(fig)

# Fig 2 period stacked bar with labels
fig, ax = plt.subplots(figsize=(9, 4.6))
bottom = np.zeros(len(per))
cmap = plt.get_cmap('tab20')
for i, c in enumerate(tcols):
    ax.bar(per.index, per[c] * 100, bottom=bottom, color=cmap(i), width=0.65, label=names[i])
    bottom += per[c].values * 100
ax.set_ylabel('평균 토픽 비중(%)'); ax.set_title('시기별 토픽 구성')
ax.legend(ncol=2, fontsize=7.5, loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig2_period_topic_share.png'), dpi=220); plt.close(fig)

# Fig 3 small multiples of yearly shares (3-yr moving avg) with trend type
yr3 = yr.rolling(3, center=True, min_periods=1).mean()
fig, axes = plt.subplots(4, 4, figsize=(11, 8.5), sharex=True)
for t in range(1, K + 1):
    ax = axes[(t - 1) // 4][(t - 1) % 4]
    row = tr[tr.topic == t].iloc[0]
    col = {'hot': 'tab:red', 'cold': 'tab:blue'}.get(row.type, 'gray')
    ax.fill_between(yr3.index, 0, yr3[f'T{t}'] * 100, color=col, alpha=0.25)
    ax.plot(yr3.index, yr3[f'T{t}'] * 100, color=col, lw=1.6)
    tag = {'hot': ' ▲부상', 'cold': ' ▼쇠퇴'}.get(row.type, '')
    ax.set_title(f'T{t} {LABELS[t]}{tag}', fontsize=9)
    ax.text(0.02, 0.9, f'ρ={row.spearman_rho:.2f}, p={row.p_value:.3f}', transform=ax.transAxes, fontsize=7, va='top')
    ax.set_ylim(0, max(30, yr3[f'T{t}'].max() * 100 * 1.15)); ax.tick_params(labelsize=8)
for ax in axes[3]:
    ax.set_xlabel('연도', fontsize=8)
for j in range(K, 16):
    axes[j // 4][j % 4].axis('off')
fig.suptitle('토픽별 연도 비중 추이(3년 이동평균, %)와 추세 검정', fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig3_topic_trends.png'), dpi=220); plt.close(fig)

# Fig 4 network: rebuild from tokens to draw
tokens = json.load(open(os.path.join(RES, 'tokens.json'), encoding='utf-8'))
docs = pd.read_csv(os.path.join(RES, 'corpus_docs.csv'))
from collections import Counter
STOP2 = set(open(os.path.join(BASE, 'analyze.py'), encoding='utf-8').read().split("STOP2 = set('''")[1].split("'''")[0].split())
toks = [[w for w in tokens[str(s)] if w not in STOP2] for s in docs.srno]
cnt = Counter(w for t in toks for w in t)
docfreq = Counter(w for t in toks for w in set(t))
NTOP = 70
top = [w for w, _ in cnt.most_common(NTOP)]
idx = {w: i for i, w in enumerate(top)}
co = np.zeros((NTOP, NTOP))
for t in toks:
    p = sorted({idx[w] for w in set(t) if w in idx})
    for a in range(len(p)):
        for b in range(a + 1, len(p)):
            co[p[a], p[b]] += 1
co += co.T
assoc = np.zeros_like(co)
for i in range(NTOP):
    for j in range(NTOP):
        if i != j:
            u = docfreq[top[i]] + docfreq[top[j]] - co[i, j]
            assoc[i, j] = co[i, j] / u if u > 0 else 0
G = nx.Graph()
for w in top:
    G.add_node(w)
for i in range(NTOP):
    for j in np.argsort(-assoc[i])[:4]:
        if assoc[i, j] > 0:
            G.add_edge(top[i], top[j], weight=float(assoc[i, j]))
comms = sorted(nx.community.louvain_communities(G, weight='weight', seed=42), key=len, reverse=True)
cmap_ = {w: ci for ci, c in enumerate(comms) for w in c}
CN = ['C1 개발·토지·인허가', 'C2 재정·주택·보증', 'C3 안전·정비·검사', 'C4 건설공사 집행', 'C5 계약·민간참여']
fig, ax = plt.subplots(figsize=(10.5, 9))
pos = nx.spring_layout(G, k=0.95, seed=42, weight='weight', iterations=300)
colors = plt.get_cmap('tab10')
wmax = max(d['weight'] for _, _, d in G.edges(data=True))
nx.draw_networkx_edges(G, pos, alpha=0.3, width=[0.3 + 3 * G[u][v]['weight'] / wmax for u, v in G.edges], ax=ax)
for ci, c in enumerate(comms):
    nx.draw_networkx_nodes(G, pos, nodelist=list(c), node_size=[180 + 14 * math.sqrt(cnt[w]) for w in c], node_color=[colors(ci)], alpha=0.9, ax=ax, label=CN[ci] if ci < len(CN) else f'C{ci + 1}')
nx.draw_networkx_labels(G, pos, font_size=8.5, font_family=plt.rcParams['font.family'], ax=ax)
leg = ax.legend(loc='lower left', fontsize=9, frameon=True, markerscale=0.25, labelspacing=0.9, borderpad=1.0)
ax.set_title('핵심 키워드 동시출현 네트워크(상위 70개 명사, 문서 수준 Jaccard 연관도, kNN=4, Louvain 군집)', fontsize=10); ax.axis('off')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig4_network.png'), dpi=220); plt.close(fig)

# Fig 5 audit type by period
docs['audKnd'] = docs['audKnd'].fillna('미분류')
ct = pd.crosstab(docs.period, docs.audKnd)
ct = ct[[c for c in ['특정사안', '국민제안', '성과', '기관정기', '확인출장', '재무', '기타(기타)', '기타(서면)', '기타(손망실)', '미분류'] if c in ct.columns]]
share = ct.div(ct.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(8, 4))
bottom = np.zeros(len(share))
for i, c in enumerate(share.columns):
    ax.bar(share.index, share[c], bottom=bottom, width=0.6, label=c, color=plt.get_cmap('tab10')(i))
    bottom += share[c].values
ax.set_ylabel('비중(%)'); ax.set_title('시기별 감사종류 구성')
ax.legend(ncol=2, fontsize=8, loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig5_audit_type.png'), dpi=220); plt.close(fig)
share.round(1).to_csv(os.path.join(RES, 'period_audit_type_share.csv'), encoding='utf-8-sig')
ct.to_csv(os.path.join(RES, 'period_audit_type_counts.csv'), encoding='utf-8-sig')
print('figures done', {ci: len(c) for ci, c in enumerate(comms)})
