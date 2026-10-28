"""Reproduce published tables and period keyword analysis; optionally refit LDA.

python run_analysis.py
python run_analysis.py --refit-lda
python run_analysis.py --search-k --refit-lda
No raw audit reports are required when using the published noun tokens.
"""
from pathlib import Path
from collections import Counter
import argparse, gzip, json, os, shutil, subprocess, sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PERIODS = ['P1 2003-2012','P2 2013-2017','P3 2018-2021','P4 2022-2026']

def save(df, name, out, index=False):
    df.to_csv(out/name, index=index, encoding='utf-8-sig')

def load_inputs():
    docs = pd.read_csv(ROOT/'published_results/corpus_docs.csv', dtype={'srno':str})
    docs = docs.sort_values(['year','srno']).reset_index(drop=True)
    with gzip.open(ROOT/'published_results/tokens.json.gz','rt',encoding='utf8') as f:
        tokens=json.load(f)
    assert docs.srno.is_unique and set(docs.srno)<=set(tokens), 'Token/metadata mismatch'
    assert len(docs)==239, 'Published corpus should contain 239 documents'
    return docs, {s:tokens[s] for s in docs.srno}

def tables(docs, out):
    from scipy.stats import spearmanr, chi2_contingency
    dt=pd.read_csv(out/'doc_topics.csv',dtype={'srno':str})
    assert set(dt.srno)==set(docs.srno) and dt.srno.is_unique
    cols=[f'T{i}' for i in range(1,15)]
    assert np.allclose(dt[cols].sum(axis=1),1,atol=1e-5)
    means=dt.groupby('period')[cols].mean().reindex(PERIODS)
    means.loc['ALL']=dt[cols].mean()
    save(means,'period_topic_share.csv',out,True)
    yr=dt.groupby('year')[cols].mean()
    save(yr,'year_topic_share.csv',out,True)
    trend=[]
    for c in cols:
        rho,p=spearmanr(yr.index,yr[c])
        trend.append({'topic':int(c[1:]),'mean_share':float(dt[c].mean()),'spearman_rho':rho,'p_value':p,
            'early_share':means.loc[PERIODS[0],c],'late_share':means.loc[PERIODS[-1],c],
            'type':'hot' if rho>0 and p<.05 else ('cold' if rho<0 and p<.05 else 'stable')})
    save(pd.DataFrame(trend),'topic_trends.csv',out)
    mapping=json.loads((ROOT/'published_results/domain_map.json').read_text(encoding='utf8'))
    drows=[]; dy=pd.DataFrame(index=yr.index)
    for domain,topics in mapping.items():
        tc=[f'T{i}' for i in topics]; y=yr[tc].sum(axis=1)
        rho,p=spearmanr(y.index,y)
        row={'domain':domain,'topics':', '.join(tc)}
        row.update({period:round(float(means.loc[period,tc].sum())*100,1) for period in PERIODS+['ALL']})
        row.update(rho=round(float(rho),2),p=round(float(p),3));drows.append(row);dy[domain]=y
    save(pd.DataFrame(drows),'domain_period_share.csv',out)
    save(dy,'domain_year_share.csv',out,True)
    save(docs.groupby('year').size().rename('n').to_frame(),'corpus_by_year.csv',out,True)
    ct=pd.crosstab(docs.period,docs.audKnd.fillna('미분류')).reindex(PERIODS,fill_value=0)
    save(ct,'period_audit_type_counts.csv',out,True)
    save(ct.div(ct.sum(axis=1),axis=0)*100,'period_audit_type_share.csv',out,True)
    dom=pd.crosstab(dt.period,dt[cols].idxmax(axis=1)).reindex(PERIODS,fill_value=0)
    chi,p,dof,exp=chi2_contingency(dom)
    (out/'dominant_topic_chi_square.json').write_text(json.dumps({'chi2':float(chi),'p':float(p),'dof':int(dof),
        'cramers_v':float(np.sqrt(chi/(dom.values.sum()*(min(dom.shape)-1)))),
        'expected_cells_below_5':int((exp<5).sum()),'note':'Exploratory, no multiplicity correction; dominant topics discard within-document mixtures.'},indent=2),encoding='utf8')
    save(dom,'period_dominant_counts.csv',out,True)

def build_network(toks, terms, seed=42):
    import networkx as nx
    count=Counter(w for t in toks for w in t)
    df=Counter(w for t in toks for w in set(t))
    ix={w:i for i,w in enumerate(terms)}; n=len(terms)
    co=np.zeros((n,n))
    for t in toks:
        present=sorted({ix[w] for w in t if w in ix})
        for a,i in enumerate(present):
            for j in present[a+1:]: co[i,j]+=1
    co+=co.T; assoc=np.zeros_like(co)
    for i in range(n):
        for j in range(n):
            union=df[terms[i]]+df[terms[j]]-co[i,j]
            if i!=j and union:assoc[i,j]=co[i,j]/union
    g=nx.Graph();g.add_nodes_from(terms)
    for i in range(n):
        # Same numpy argsort convention as the upstream code. Ties may vary by numpy version.
        for j in np.argsort(-assoc[i])[:4]:
            if i!=j and assoc[i,j]>0:g.add_edge(terms[i],terms[j],weight=float(assoc[i,j]),co=int(co[i,j]))
    communities=sorted(nx.community.louvain_communities(g,weight='weight',seed=seed),key=len,reverse=True) if g.number_of_edges() else [{w} for w in terms]
    community={w:i for i,c in enumerate(communities) for w in c}
    deg=nx.degree_centrality(g);btw=nx.betweenness_centrality(g,weight=None)
    nodes=pd.DataFrame([{'term':w,'freq':count[w],'doc_freq':df[w],'neighbors':int(g.degree[w]),
        'degree':deg[w] if n>1 else 0,'betweenness':btw[w],'community':community[w]} for w in terms])
    nodes=nodes.sort_values(['degree','freq','term'],ascending=[False,False,True])
    edges=pd.DataFrame([{'source':a,'target':b,'jaccard':d['weight'],'co_documents':d['co']} for a,b,d in g.edges(data=True)])
    stats={'nodes':n,'edges':g.number_of_edges(),'density':nx.density(g),'n_communities':len(communities),
        'modularity':nx.community.modularity(g,communities,weight='weight') if g.number_of_edges() else None,'n_documents':len(toks)}
    return g,nodes,edges,stats

def keywords(docs,tokens,out):
    from sklearn.feature_extraction.text import TfidfVectorizer
    toks=[tokens[s] for s in docs.srno]
    count=Counter(w for t in toks for w in t)
    common=[w for w,_ in count.most_common(70)]
    save(pd.DataFrame(count.most_common(300),columns=['term','freq']),'term_freq_top300.csv',out)
    graphs={}; allnodes=[]; allstats=[]; top=[]
    for mode in ['common','period']:
        for period in PERIODS:
            sub=[tokens[s] for s in docs.loc[docs.period==period,'srno']]
            cnt=Counter(w for t in sub for w in t)
            terms=common if mode=='common' else [w for w,_ in cnt.most_common(70)]
            g,nodes,edges,stats=build_network(sub,terms)
            key=period[:2]+'_'+mode
            save(nodes,f'network_nodes_{key}.csv',out)
            save(edges,f'network_edges_{key}.csv',out)
            nodes.insert(0,'period',period);nodes.insert(0,'node_mode',mode);allnodes.append(nodes)
            allstats.append(dict(node_mode=mode,period=period,**stats));graphs[key]=(g,nodes)
            for rank,row in enumerate(nodes.head(10).to_dict('records'),1):top.append(dict(rank=rank,**row))
            save(pd.DataFrame(cnt.most_common(300),columns=['term','freq']),f'frequency_{period[:2]}.csv',out)
    save(pd.concat(allnodes),'period_network_nodes.csv',out)
    save(pd.DataFrame(allstats),'period_network_stats.csv',out)
    save(pd.DataFrame(top),'period_network_top10.csv',out)
    vec=TfidfVectorizer(token_pattern=r'\S+',sublinear_tf=True,min_df=5,max_df=.7)
    x=vec.fit_transform([' '.join(t) for t in toks]);vocab=np.array(vec.get_feature_names_out())
    overall=np.asarray(x.mean(axis=0)).ravel(); rows=[]
    for p in PERIODS:
        mask=(docs.period==p).to_numpy();mean=np.asarray(x[mask].mean(axis=0)).ravel()
        df=np.asarray((x[mask]>0).sum(axis=0)).ravel();ratio=(mean+1e-6)/(overall+1e-6)
        for j in np.argsort(-mean)[:30]:rows.append({'period':p,'term':vocab[j],'mean_tfidf':mean[j],
            'relative_to_all':ratio[j],'document_frequency':int(df[j])})
    save(pd.DataFrame(rows),'period_tfidf_top30.csv',out)
    return graphs

def figures(out,graphs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import networkx as nx
    names={f.name for f in font_manager.fontManager.ttflist}
    for candidate in ['Malgun Gothic','AppleGothic','Noto Sans CJK KR','Noto Sans CJK JP','NanumGothic']:
        if candidate in names:plt.rcParams['font.family']=candidate;break
    plt.rcParams['axes.unicode_minus']=False
    dest=out/'figures';dest.mkdir(exist_ok=True)
    def finish(fig,name):
        fig.tight_layout();fig.savefig(dest/name,dpi=180);plt.close(fig)
    c=pd.read_csv(out/'coherence.csv');fig,ax=plt.subplots(figsize=(6,4));ax.plot(c.K,c.c_v,'o-');ax.axvline(14,color='gray',ls='--');ax.set(xlabel='K',ylabel='c_v');finish(fig,'fig1_coherence.png')
    d=pd.read_csv(out/'domain_period_share.csv').set_index('domain')[PERIODS].T
    fig,ax=plt.subplots(figsize=(10,5));d.plot.bar(stacked=True,ax=ax);ax.set(ylabel='%',xlabel='');ax.legend(fontsize=7,bbox_to_anchor=(1,1));finish(fig,'fig2_domains.png')
    yr=pd.read_csv(out/'year_topic_share.csv',index_col=0);tr=pd.read_csv(out/'topic_trends.csv').set_index('topic')
    labels=json.loads((ROOT/'published_results/topic_labels.json').read_text(encoding='utf8'))
    fig,axes=plt.subplots(4,4,figsize=(13,10));sm=yr.rolling(3,center=True,min_periods=1).mean()
    for i,ax in enumerate(axes.flat,1):
        if i>14:ax.axis('off');continue
        ax.plot(sm.index,sm[f'T{i}']*100);ax.set_title(f'T{i} '+labels[str(i)],fontsize=8)
        ax.text(.02,.9,f"rho={tr.loc[i,'spearman_rho']:.2f}, p={tr.loc[i,'p_value']:.3f}",transform=ax.transAxes,fontsize=7)
    fig.suptitle('Topic shares: moving average over three observed years');finish(fig,'fig3_topic_trends.png')
    for mode in ['common','period']:
        fig,axes=plt.subplots(2,2,figsize=(16,14))
        for period,ax in zip(PERIODS,axes.flat):
            g,nodes=graphs[period[:2]+'_'+mode];lookup=nodes.set_index('term')
            active=[w for w in g if g.degree[w]>0];isolated=[w for w in g if g.degree[w]==0]
            pos=nx.spring_layout(g.subgraph(active),seed=42,k=.55,weight='weight',iterations=300)
            # Keep isolated keywords visible without allowing them to compress the connected graph.
            for i,w in enumerate(isolated):pos[w]=np.array([-.9+1.8*(i+1)/(len(isolated)+1),-1.35])
            nx.draw_networkx_edges(g,pos,alpha=.2,ax=ax)
            nx.draw_networkx_nodes(g,pos,node_size=[60+lookup.loc[w,'degree']*1500 for w in g],node_color=[lookup.loc[w,'community'] for w in g],cmap='tab10',ax=ax)
            nx.draw_networkx_labels(g,pos,font_size=7,font_family=plt.rcParams['font.family'],ax=ax)
            ax.set_title(period);ax.axis('off')
        finish(fig,f'fig4_period_network_{mode}.png')
    ct=pd.read_csv(out/'period_audit_type_share.csv',index_col=0);fig,ax=plt.subplots(figsize=(10,5));ct.plot.bar(stacked=True,ax=ax);ax.set(ylabel='%',xlabel='');ax.legend(fontsize=8,bbox_to_anchor=(1,1));finish(fig,'fig5_audit_types.png')
    dy=pd.read_csv(out/'domain_year_share.csv',index_col=0);fig,ax=plt.subplots(figsize=(10,5));dy.rolling(3,center=True,min_periods=1).mean().plot.area(ax=ax);ax.legend(fontsize=7,bbox_to_anchor=(1,1));finish(fig,'fig6_domain_trends.png')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--refit-lda',action='store_true');ap.add_argument('--search-k',action='store_true')
    ap.add_argument('--output',type=Path,default=ROOT/'output');ap.add_argument('--no-figures',action='store_true');args=ap.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    docs,tokens=load_inputs()
    for p in (ROOT/'published_results').iterdir():
        if p.suffix in ['.csv','.json']:shutil.copy2(p,out/p.name)
    (out/'tokens.json').write_text(json.dumps(tokens,ensure_ascii=False),encoding='utf8')
    if args.refit_lda or args.search_k:
        env=dict(os.environ,BAI_RESULTS=str(out),PYTHONIOENCODING='utf-8')
        if args.search_k:subprocess.run([sys.executable,str(ROOT/'code/analyze.py'),'--kmin','6','--kmax','20','--kstep','2'],check=True,env=env)
        # The search winner is not automatically the manuscript model. Final paper uses K=14.
        subprocess.run([sys.executable,str(ROOT/'code/analyze.py'),'--k','14','--seed','42'],check=True,env=env)
    tables(docs,out);graphs=keywords(docs,tokens,out)
    if not args.no_figures:figures(out,graphs)
    stats={'n_documents':len(docs),'period_counts':docs.groupby('period').size().to_dict(),'n_tokens':sum(map(len,tokens.values())),
        'token_count_matches_metadata':all(len(tokens[r.srno])==r.n_tokens for r in docs.itertuples()),'refitted_lda':args.refit_lda or args.search_k,
        'network_modes':{'common':'same overall top 70 terms; degree denominator 69','period':'top 70 terms selected separately in each period'},
        'selection_note':'Selected published reports, not prevalence of all implementation defects.',
        'published_commit':json.loads((ROOT/'provenance.json').read_text(encoding='utf8'))['commit']}
    (out/'run_summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(stats,ensure_ascii=False,indent=2));print('Output:',out)

if __name__=='__main__':main()
