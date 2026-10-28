"""Grayscale (print-friendly) figures. Writes figures/fig*.png (overwrites colour versions; colour copies kept in figures_color/)."""
import os, json, math
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import networkx as nx
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(BASE, 'results'), os.path.join(BASE, 'figures')
fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

if not os.path.exists(fp):
    fp = font_manager.findfont('Malgun Gothic' if os.name == 'nt' else 'DejaVu Sans')
font_manager.fontManager.addfont(fp)
plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['hatch.linewidth'] = 0.6

LABELS = json.load(open(os.path.join(RES, 'topic_labels.json'), encoding='utf-8'))
LABELS = {int(k): v for k, v in LABELS.items()}
K = len(LABELS)
coh = pd.read_csv(os.path.join(RES, 'coherence.csv'))
yr = pd.read_csv(os.path.join(RES, 'year_topic_share.csv'), index_col=0)
tr = pd.read_csv(os.path.join(RES, 'topic_trends.csv'))
GRAYS6 = ['#8c8c8c', '#ffffff', '#bfbfbf', '#ffffff', '#e3e3e3', '#f7f7f7']
HATCH6 = ['', '///', '', 'xxx', '', '...']

# Fig 1 coherence
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(coh.K, coh.c_v, marker='o', color='black', label='c_v')
ax.axvline(14, ls='--', c='gray', lw=1)
ax.set_xlabel('토픽 수 (K)'); ax.set_ylabel('일관성 c_v'); ax.set_title('토픽 수별 일관성 점수')
ax2 = ax.twinx(); ax2.plot(coh.K, coh.u_mass, marker='s', ls=':', color='dimgray', label='u_mass'); ax2.set_ylabel('u_mass')
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc='lower right', fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig1_coherence.png'), dpi=220); plt.close(fig)

# Fig 3 small multiples
yr3 = yr.rolling(3, center=True, min_periods=1).mean()
fig, axes = plt.subplots(4, 4, figsize=(11, 8.5), sharex=True)
for t in range(1, K + 1):
    ax = axes[(t - 1) // 4][(t - 1) % 4]
    row = tr[tr.topic == t].iloc[0]
    y = yr3[f'T{t}'] * 100
    if row.type == 'hot':
        ax.fill_between(yr3.index, 0, y, color='#9a9a9a'); tag = ' ▲부상'
    elif row.type == 'cold':
        ax.fill_between(yr3.index, 0, y, facecolor='white', hatch='////', edgecolor='#7a7a7a', lw=0); tag = ' ▽쇠퇴'
    else:
        ax.fill_between(yr3.index, 0, y, color='#e0e0e0'); tag = ''
    ax.plot(yr3.index, y, color='#333333', lw=1.2)
    ax.set_title(f'T{t} {LABELS[t]}{tag}', fontsize=9)
    ax.text(0.02, 0.9, f'ρ={row.spearman_rho:.2f}, p={row.p_value:.3f}', transform=ax.transAxes, fontsize=7, va='top')
    ax.set_ylim(0, max(30, y.max() * 1.15)); ax.tick_params(labelsize=8)
for ax in axes[3]:
    ax.set_xlabel('연도', fontsize=8)
for j in range(K, 16):
    axes[j // 4][j % 4].axis('off')
leg = [Patch(facecolor='#9a9a9a', label='부상(p<.05)'), Patch(facecolor='white', hatch='////', edgecolor='#7a7a7a', label='쇠퇴(p<.05)'), Patch(facecolor='#e0e0e0', label='안정')]
axes[3][2].legend(handles=leg, loc='center', fontsize=9, frameon=False)
fig.suptitle('토픽별 연도 비중 추이(3년 이동평균, %)와 추세 검정', fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig3_topic_trends.png'), dpi=220); plt.close(fig)

# Fig 6 domain stacked area (grayscale + hatch)
ydom = pd.read_csv(os.path.join(RES, 'domain_year_share.csv'), index_col=0)
y3 = ydom.rolling(3, center=True, min_periods=1).mean()
fig, ax = plt.subplots(figsize=(9, 4.6))
polys = ax.stackplot(y3.index, [y3[c] for c in y3.columns], colors=GRAYS6, edgecolor='black', linewidth=0.4)
for p, h in zip(polys, HATCH6):
    p.set_hatch(h)
ax.set_xlim(2003, 2026); ax.set_ylim(0, 100); ax.set_ylabel('정책 영역별 토픽 비중(%, 3년 이동평균)'); ax.set_xlabel('연도'); ax.set_title('주택·도시·국토 정책 영역별 감사 쟁점 비중의 변화')
handles = [Patch(facecolor=GRAYS6[i], hatch=HATCH6[i], edgecolor='black', label=c) for i, c in enumerate(y3.columns)]
ax.legend(handles=handles[::-1], loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig6_domain_share.png'), dpi=220); plt.close(fig)

# Fig 5 audit type stacked bar
docs = pd.read_csv(os.path.join(RES, 'corpus_docs.csv')); docs['audKnd'] = docs['audKnd'].fillna('미분류')
ct = pd.crosstab(docs.period, docs.audKnd)
order = [c for c in ['특정사안', '국민제안', '성과', '기관정기', '확인출장'] if c in ct.columns]
other = [c for c in ct.columns if c not in order]
ct2 = ct[order].copy(); ct2['기타'] = ct[other].sum(axis=1)
share = ct2.div(ct2.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(8, 4))
bottom = np.zeros(len(share))
for i, c in enumerate(share.columns):
    ax.bar(share.index, share[c], bottom=bottom, width=0.6, label=c, color=GRAYS6[i], hatch=HATCH6[i], edgecolor='black', linewidth=0.5)
    bottom += share[c].values
ax.set_ylabel('비중(%)'); ax.set_title('시기별 감사종류 구성')
ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig5_audit_type.png'), dpi=220); plt.close(fig)

# Fig 4 network: shapes + grays per community
tokens = json.load(open(os.path.join(RES, 'tokens.json'), encoding='utf-8'))
STOP2 = set(open(os.path.join(BASE, 'code', 'analyze.py'), encoding='utf-8').read().split("STOP2 = set('''")[1].split("'''")[0].split())
toks = [[w for w in tokens[str(s)] if w not in STOP2] for s in docs.srno]
cnt = Counter(w for t in toks for w in t); docfreq = Counter(w for t in toks for w in set(t))
NTOP = 70; top = [w for w, _ in cnt.most_common(NTOP)]; idx = {w: i for i, w in enumerate(top)}
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
            u = docfreq[top[i]] + docfreq[top[j]] - co[i, j]; assoc[i, j] = co[i, j] / u if u > 0 else 0
G = nx.Graph(); G.add_nodes_from(top)
for i in range(NTOP):
    for j in np.argsort(-assoc[i])[:4]:
        if assoc[i, j] > 0: G.add_edge(top[i], top[j], weight=float(assoc[i, j]))
comms = sorted(nx.community.louvain_communities(G, weight='weight', seed=42), key=len, reverse=True)
CN = ['C1 개발·토지·인허가', 'C2 재정·주택·보증', 'C3 안전·정비·검사', 'C4 건설공사 집행', 'C5 계약·민간참여']
SHAPES = ['o', 's', '^', 'D', 'v']; FILLS = ['#2b2b2b', '#ffffff', '#8c8c8c', '#ffffff', '#c8c8c8']; FONTC = ['white', 'black', 'black', 'black', 'black']
FILLS = ['#bdbdbd', '#ffffff', '#e0e0e0', '#ffffff', '#f4f4f4']
cmap_ = {w: ci for ci, c in enumerate(comms) for w in c}
# same layout as the colour version (spring layout), grayscale fills + marker shapes
pos = nx.spring_layout(G, k=0.95, seed=42, weight='weight', iterations=300)
wmax = max(d['weight'] for _, _, d in G.edges(data=True))
fig, ax = plt.subplots(figsize=(10.5, 9))
nx.draw_networkx_edges(G, pos, alpha=0.35, edge_color='#8a8a8a', width=[0.3 + 3 * G[u][v]['weight'] / wmax for u, v in G.edges], ax=ax)
for ci, c in enumerate(comms):
    nodes = list(c)
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=[180 + 14 * math.sqrt(cnt[w]) for w in nodes], node_color=FILLS[ci], node_shape=SHAPES[ci], edgecolors='#444444', linewidths=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={w: w for w in nodes}, font_size=8.5, font_color='black', font_family=plt.rcParams['font.family'], ax=ax)
handles = [Line2D([0], [0], marker=SHAPES[i], color='w', markerfacecolor=FILLS[i], markeredgecolor='#444444', markersize=10, label=CN[i]) for i in range(len(comms))]
ax.legend(handles=handles, loc='lower left', fontsize=9, frameon=True)
ax.set_title('핵심 키워드 동시출현 네트워크(상위 70개 명사, 문서 수준 Jaccard 연관도, kNN=4, Louvain 군집)', fontsize=10); ax.axis('off')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig4_network.png'), dpi=220); plt.close(fig)

# Fig 2 period topic stacked bar (14 topics) grayscale with hatches
per = pd.read_csv(os.path.join(RES, 'period_topic_share.csv'), index_col=0).drop('ALL')
tcols = [f'T{t}' for t in range(1, K + 1)]
G14 = [plt.get_cmap('Greys')(0.05 + 0.55 * i / 13) for i in range(14)]
H14 = ['', '//', '', 'xx', '', '..', '', '\\\\', '', '++', '', '//', '', 'xx']
fig, ax = plt.subplots(figsize=(9, 4.6))
bottom = np.zeros(len(per))
for i, c in enumerate(tcols):
    ax.bar(per.index, per[c] * 100, bottom=bottom, color=G14[i], hatch=H14[i], edgecolor='black', linewidth=0.4, width=0.65, label=f'{c} {LABELS[i + 1]}')
    bottom += per[c].values * 100
ax.set_ylabel('평균 토픽 비중(%)'); ax.set_title('시기별 토픽 구성')
ax.legend(ncol=2, fontsize=7.5, loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig2_period_topic_share.png'), dpi=220); plt.close(fig)
print('bw figures done')
