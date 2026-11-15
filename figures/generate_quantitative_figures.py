#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results'/'paper_scale'; FIG=ROOT/'figures'/'generated'; STYLE=ROOT/'figures'/'styles'/'scienceplots_ieee_vendored.mplstyle'
FIG.mkdir(parents=True,exist_ok=True)
plt.style.use(str(STYLE))

# Restrained IEEE-style semantic palette.
NAVY='#2C5B84'; BLUE='#3B82F6'; ORANGE='#F5A623'; CORAL='#EF5B6C'; PURPLE='#8A6EDB'; TEAL='#2A9D8F'; GREEN='#3A9D5D'; GRAY='#6B7280'; DARK='#222222'; LIGHT='#D9DEE5'
COLORS=[NAVY,BLUE,ORANGE,CORAL,PURPLE,TEAL]

mpl.rcParams.update({
    'figure.facecolor':'white','axes.facecolor':'white',
    'axes.edgecolor':'#3A3A3A','axes.labelcolor':DARK,'text.color':DARK,
    'xtick.color':'#3A3A3A','ytick.color':'#3A3A3A',
    'legend.handlelength':1.6,'legend.columnspacing':0.9,'legend.labelspacing':0.25,
})

def tidy(ax, grid='y'):
    for sp in ax.spines.values(): sp.set_linewidth(.5); sp.set_color('#505050')
    if grid:
        ax.grid(axis=grid,color='#CBD1D8',lw=.35,alpha=.45,zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(which='both',direction='in',top=True,right=True)

def panel(ax, label, title=None):
    ax.text(-.12,1.04,f'({label})',transform=ax.transAxes,fontsize=8.5,fontweight='bold',ha='left',va='bottom')
    if title: ax.set_title(title,pad=4,fontweight='bold')

def save(fig,name):
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight',pad_inches=.025)
    plt.close(fig)

# ---------------- Fig.3: controlled-masking comparison ----------------
mc=pd.read_csv(DATA/'fig_main_comparison.csv')
methods=['zero_fill','complete_case','train_mean','train_median','knn','random_forest_imputation']
labels={'zero_fill':'Zero-fill','complete_case':'Complete-case','train_mean':'Train mean','train_median':'Train median','knn':'KNN','random_forest_imputation':'RF'}
colors=dict(zip(methods,[GRAY,PURPLE,ORANGE,CORAL,BLUE,TEAL]))
markers={'zero_fill':'o','complete_case':'s','train_mean':'^','train_median':'v','knn':'D','random_forest_imputation':'P'}
fig,axs=plt.subplots(2,2,figsize=(7.0,5.25))
fig.subplots_adjust(left=.078,right=.982,bottom=.092,top=.895,wspace=.38,hspace=.37)
# Common method legend for the two top-row point-error panels: outside data regions.
method_handles=[]
for m in methods:
    method_handles.append(Line2D([0],[0],color=colors[m],marker=markers[m],lw=.95,ms=3.0,label=labels[m]))
fig.legend(handles=method_handles,ncol=6,loc='upper center',bbox_to_anchor=(.53,.995),fontsize=5.55,frameon=False,
           handlelength=1.35,columnspacing=.85,handletextpad=.35)
for ax,ds,lab,title in [(axs[0,0],'FreeInference','a','FreeInference: point error'),(axs[0,1],'SyFI','b','TraceLab: point error')]:
    for m in methods:
        z=mc[(mc.dataset==ds)&(mc.output_type=='point')&(mc.method==m)]
        g=z.groupby('q')['median_relative_error'].agg(['median','min','max']).reset_index().sort_values('q')
        x=g.q*100; y=g['median']*100
        ax.plot(x,y,marker=markers[m],color=colors[m],lw=.95,ms=3.0)
        ax.fill_between(x,g['min']*100,g['max']*100,color=colors[m],alpha=.07,lw=0)
    ax.set_yscale('log'); ax.set_xticks([10,20,30]); ax.set_xlabel('Masked target fraction (%)'); ax.set_ylabel('Median settlement error (%)')
    tidy(ax); panel(ax,lab,title)
# c: explicitly separate prediction error (solid) and unsupported-point rate (dashed).
ax=axs[1,0]
handles_c=[]
for ds,c,mk,disp in [('FreeInference',NAVY,'o','FreeInference'),('SyFI',ORANGE,'s','TraceLab')]:
    z=mc[(mc.dataset==ds)&(mc.output_type=='point')&(mc.method=='random_forest_imputation')]
    e=z.groupby('q').median(numeric_only=True).sort_index()
    ax.plot(e.index*100,e.median_relative_error*100,color=c,marker=mk,lw=1.05,ms=3.0)
    handles_c.append(Line2D([0],[0],color=c,marker=mk,lw=1.05,ms=3.0,label=f'{disp}: MedRE'))
ax.set_xlabel('Masked target fraction (%)'); ax.set_ylabel('RF MedRE (%)',labelpad=2); tidy(ax); panel(ax,'c','Low prediction error, non-final evidence')
ax2=ax.twinx()
for ds,c,mk,disp in [('FreeInference',NAVY,'o','FreeInference'),('SyFI',ORANGE,'s','TraceLab')]:
    z=mc[(mc.dataset==ds)&(mc.output_type=='point')&(mc.method=='random_forest_imputation')]
    e=z.groupby('q').median(numeric_only=True).sort_index()
    ax2.plot(e.index*100,e.unsupported_point_rate*100,color=c,marker=mk,lw=.75,ms=2.7,ls='--',alpha=.82)
    handles_c.append(Line2D([0],[0],color=c,marker=mk,lw=.75,ls='--',ms=2.7,label=f'{disp}: unsupported'))
ax2.set_ylim(98.5,100.15); ax2.set_ylabel('Unsupported point rate (%)',labelpad=1); ax2.tick_params(labelsize=6.3,pad=1,direction='in')
# Place legend in the visually empty center-left region, away from both solid and dashed curves.
ax.legend(handles=handles_c,loc='center left',bbox_to_anchor=(.02,.56),fontsize=5.05,frameon=False,
          handlelength=1.55,labelspacing=.25,borderaxespad=0)
# d: interval semantics. Explicit four-combination legend; no hidden size encoding.
ax=axs[1,1]
combo_handles=[]
for method,c,mlab in [('chained_equations',ORANGE,'Chained eq.'),('EBMUS_evidence_bounds',TEAL,'Evidence bounds')]:
    for ds,mk,dlabel in [('FreeInference','o','FreeInference'),('SyFI','^','TraceLab')]:
        z=mc[(mc.output_type=='interval')&(mc.method==method)&(mc.dataset==ds)]
        ax.scatter(z.median_relative_width*100,z.coverage*100,s=24,color=c,marker=mk,alpha=.78,
                   edgecolors='white',linewidth=.35,zorder=3)
        combo_handles.append(Line2D([0],[0],marker=mk,color='none',markerfacecolor=c,markeredgecolor='white',
                                    markeredgewidth=.35,markersize=4.8,label=f'{mlab} — {dlabel}'))
ax.axhline(100,color=GRAY,lw=.55,ls='--')
ax.set_xlabel('Median interval width, $W_{rel}$ (%)'); ax.set_ylabel('True-settlement coverage (%)',labelpad=2); ax.set_ylim(0,103)
tidy(ax); panel(ax,'d','Interval width vs. coverage')
# Lower-right is empty because evidence-bound points sit near y=100 and chained-equation points occupy left/middle.
ax.legend(handles=combo_handles,ncol=1,loc='lower right',fontsize=4.95,frameon=False,
          handletextpad=.35,labelspacing=.20,borderaxespad=.35)
save(fig,'fig3_main_comparison')

# ---------------- Fig.4: refinement / scaling / structural value ----------------
er=pd.read_csv(DATA/'fig_evidence_refinement.csv'); ss=pd.read_csv(DATA/'fig_solver_scaling.csv')
semantic={'E0':'Local','E1':'+Subtree','E2':'+Attrib.','E3':'+Identity','E4':'Full'}; order=['E0','E1','E2','E3','E4']; x=np.arange(5)
fig,axs=plt.subplots(2,2,figsize=(7.0,5.15)); fig.subplots_adjust(left=.075,right=.985,bottom=.09,top=.96,wspace=.25,hspace=.34)
for ax,metric,ylabel,lab,title in [(axs[0,0],'median_Wrel','Median $W_{rel}$','a','Evidence refinement'),(axs[0,1],'final_1pct_fraction','Final@1% (%)','b','Finality under refinement')]:
    for rh,c,mk in [('low',BLUE,'o'),('medium',ORANGE,'s'),('high',CORAL,'^')]:
        z=er[er.rate_heterogeneity==rh]
        med=[]; lo=[]; hi=[]
        for cond in order:
            zz=z[z.condition==cond][metric]*(100 if metric=='final_1pct_fraction' else 1)
            med.append(zz.median()); lo.append(zz.min()); hi.append(zz.max())
        ax.plot(x,med,color=c,marker=mk,lw=1.0,ms=3.0,label=rh.capitalize())
        ax.fill_between(x,lo,hi,color=c,alpha=.08,lw=0)
    ax.set_xticks(x,[semantic[o] for o in order],rotation=13); ax.set_ylabel(ylabel)
    if metric=='final_1pct_fraction': ax.set_ylim(-2,102)
    tidy(ax); panel(ax,lab,title)
axs[0,0].legend(ncol=3,loc='upper right',fontsize=5.8,title='Rate heterogeneity',title_fontsize=6.0)
ax=axs[1,0]
for method,c,ls,label,mk in [('generic_lp',GRAY,'--','Generic LP','o'),('laminarflow',TEAL,'-','LaminarFlow','s'),('generic_milp',PURPLE,'--','Generic MILP','^'),('evidence_milp',BLUE,'-','Scoped branch','D')]:
    z=ss[(ss.method==method)&(ss.success)]
    rows=[]
    for n,zz in z.groupby('n'): rows.append((n,zz.runtime_seconds.median(),zz.runtime_seconds.quantile(.25),zz.runtime_seconds.quantile(.75)))
    g=pd.DataFrame(rows,columns=['n','med','lo','hi']).sort_values('n')
    ax.plot(g.n,g.med,color=c,ls=ls,marker=mk,lw=.95,ms=2.8,label=label); ax.fill_between(g.n,g.lo,g.hi,color=c,alpha=.06,lw=0)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Delegation-tree size $n$'); ax.set_ylabel('Lower + upper runtime (s)'); tidy(ax); panel(ax,'c','Exact solver scaling'); ax.legend(ncol=2,loc='upper left',fontsize=5.6)
ax=axs[1,1]
for method,c,label,mk in [('independent_bounds',ORANGE,'Independent bounds','o'),('laminarflow',TEAL,'Structured exact','s')]:
    z=ss[(ss.problem=='continuous_laminar')&(ss.method==method)&(ss.success)]
    rows=[]
    for n,zz in z.groupby('n'): rows.append((n,zz.relative_width.median(),zz.relative_width.quantile(.25),zz.relative_width.quantile(.75)))
    g=pd.DataFrame(rows,columns=['n','med','lo','hi']).sort_values('n')
    ax.plot(g.n,g.med,color=c,marker=mk,lw=1.05,ms=3.0,label=label); ax.fill_between(g.n,g.lo,g.hi,color=c,alpha=.08,lw=0)
ax.set_xscale('log'); ax.set_xlabel('Delegation-tree size $n$'); ax.set_ylabel('Relative interval width'); tidy(ax); panel(ax,'d','Structural tightness')
ax.annotate('2.19×',xy=(1e4,.3248),xytext=(1800,.39),fontsize=6.5,color=GRAY,arrowprops=dict(arrowstyle='->',lw=.55,color=GRAY)); ax.legend(loc='upper left',fontsize=5.8)
save(fig,'fig4_refinement_scaling')

# ---------------- Fig.5 main: compact two-interaction summary ----------------
sn=pd.read_csv(DATA/'fig_sensitivity.csv')
rate_order=['low','medium','high']; hat=[.05,.10,.20]; miss=[.05,.10,.20]; cov=[.25,.50,.75]
za=sn.groupby(['rate_heterogeneity','hidden_attribution']).median(numeric_only=True)['median_Wrel'].unstack().reindex(rate_order).reindex(columns=hat)
zb=sn.groupby(['missing_local','internal_summary_coverage']).median(numeric_only=True)['median_Wrel'].unstack().reindex(miss).reindex(columns=cov)
vmin=0; vmax=max(float(np.nanmax(za.values)),float(np.nanmax(zb.values)))
fig,axs=plt.subplots(2,1,figsize=(3.35,4.4)); fig.subplots_adjust(left=.22,right=.88,bottom=.10,top=.96,hspace=.48)
for ax,z,lab,title,xlab,ylab,xlabels,ylabels in [
    (axs[0],za,'a','Attribution × rate heterogeneity','Hidden model/rate attribution','Rate heterogeneity',['5%','10%','20%'],['Low','Medium','High']),
    (axs[1],zb,'b','Local missingness × summary coverage','Internal-summary coverage','Missing local usage',['25%','50%','75%'],['5%','10%','20%'])]:
    im=ax.imshow(z.values,origin='lower',aspect='auto',cmap='YlOrRd',vmin=vmin,vmax=vmax)
    ax.set_xticks(range(3),xlabels); ax.set_yticks(range(3),ylabels); ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    for i in range(3):
        for j in range(3):
            val=z.values[i,j]; ax.text(j,i,f'{val:.3f}',ha='center',va='center',fontsize=6.7,fontweight='bold',color='white' if val>.60*vmax else DARK)
    for sp in ax.spines.values(): sp.set_linewidth(.5)
    panel(ax,lab,title)
cax=fig.add_axes([.91,.19,.025,.64]); cb=fig.colorbar(im,cax=cax); cb.set_label('Median $W_{rel}$',fontsize=7); cb.ax.tick_params(labelsize=6.5,direction='in')
save(fig,'fig5_sensitivity')

# Supplementary full 243-cell view.
amb_order=['none','low','medium']; heat=[]
for amb in amb_order:
    z=sn[sn.ambiguity_density==amb].groupby(['rate_heterogeneity','hidden_attribution']).median(numeric_only=True)['median_Wrel'].unstack().reindex(rate_order).reindex(columns=hat); heat.append(z)
allvals=np.concatenate([z.values.ravel() for z in heat]+[zb.values.ravel()]); vmaxf=float(np.nanmax(allvals))
fig,axs=plt.subplots(2,2,figsize=(6.8,5.1)); fig.subplots_adjust(left=.10,right=.90,bottom=.10,top=.95,wspace=.30,hspace=.36)
for idx,(ax,amb,z) in enumerate(zip(axs.flat[:3],amb_order,heat)):
    im=ax.imshow(z.values,origin='lower',aspect='auto',cmap='YlOrRd',vmin=0,vmax=vmaxf)
    ax.set_xticks(range(3),['5%','10%','20%']); ax.set_yticks(range(3),['Low','Medium','High']); ax.set_xlabel('Hidden attribution'); ax.set_ylabel('Rate heterogeneity')
    for i in range(3):
        for j in range(3):
            val=z.values[i,j]; ax.text(j,i,f'{val:.3f}',ha='center',va='center',fontsize=6.5,color='white' if val>.60*vmaxf else DARK)
    panel(ax,chr(ord('a')+idx),f'Ambiguity: {amb}')
ax=axs[1,1]; im=ax.imshow(zb.values,origin='lower',aspect='auto',cmap='YlOrRd',vmin=0,vmax=vmaxf); ax.set_xticks(range(3),['25%','50%','75%']); ax.set_yticks(range(3),['5%','10%','20%']); ax.set_xlabel('Internal-summary coverage'); ax.set_ylabel('Missing local usage')
for i in range(3):
    for j in range(3):
        val=zb.values[i,j]; ax.text(j,i,f'{val:.3f}',ha='center',va='center',fontsize=6.5,color='white' if val>.60*vmaxf else DARK)
panel(ax,'d','Coverage interaction')
cax=fig.add_axes([.925,.19,.018,.64]); cb=fig.colorbar(im,cax=cax); cb.set_label('Median $W_{rel}$',fontsize=7); cb.ax.tick_params(labelsize=6.5,direction='in')
save(fig,'figS_sensitivity_full')

# ---------------- Fig.6: natural trace and cross-provider replication ----------------
nat=pd.read_csv(DATA/'fig_freeinference_natural.csv'); tol=pd.read_csv(DATA/'paper_freeinference_tolerance_curve.csv'); rg=pd.read_csv(DATA/'fig_real_trace_generalization.csv'); nat['date']=pd.to_datetime(nat.week.str.split('_').str[0])
fig,axs=plt.subplots(2,2,figsize=(7.0,5.1)); fig.subplots_adjust(left=.08,right=.985,bottom=.10,top=.96,wspace=.25,hspace=.34)
ax=axs[0,0]; ax.plot(nat.date,nat.Wrel*100,marker='o',ms=2.9,lw=1.0,color=TEAL,label='$W_{rel}$'); ax.plot(nat.date,nat.natural_null_fraction*100,marker='s',ms=2.7,lw=.9,color=ORANGE,label='Natural-null fraction'); ax.set_ylabel('Percent (%)'); ax.tick_params(axis='x',rotation=28); tidy(ax); panel(ax,'a','16-week natural evidence state'); ax.legend(ncol=2,loc='upper right',fontsize=5.7)
ax=axs[0,1]; sizes=13+30*(nat.requests-nat.requests.min())/(nat.requests.max()-nat.requests.min()); ax.scatter(nat.natural_null_fraction*100,nat.Wrel*100,s=sizes,color=ORANGE,alpha=.75,edgecolor='white',linewidth=.35,zorder=3); coef=np.polyfit(nat.natural_null_fraction,nat.Wrel,1); xx=np.linspace(nat.natural_null_fraction.min(),nat.natural_null_fraction.max(),100); ax.plot(xx*100,np.polyval(coef,xx)*100,color=GRAY,lw=.75,ls='--'); ax.text(.04,.94,'Spearman $\\rho=0.929$\n$p=1.94\\times10^{-7}$',transform=ax.transAxes,va='top',fontsize=6.4); ax.set_xlabel('Natural-null fraction (%)'); ax.set_ylabel('Weekly $W_{rel}$ (%)'); tidy(ax); panel(ax,'b','Natural missingness vs. non-finality')
ax=axs[1,0]
for col,c,label,mk in [('finality_fraction',TEAL,'All workflows','o'),('nested_only_finality_fraction',NAVY,'Nested/delegated','s')]:
    rows=[]
    for tt,z in tol.groupby('tolerance'):
        v=z[col]*100; rows.append((tt*100,v.median(),v.quantile(.25),v.quantile(.75)))
    g=pd.DataFrame(rows,columns=['tol','med','lo','hi']).sort_values('tol'); ax.plot(g.tol,g.med,color=c,marker=mk,lw=1.0,ms=2.9,label=label); ax.fill_between(g.tol,g.lo,g.hi,color=c,alpha=.08,lw=0)
ax.set_xlabel('Finality tolerance (%)'); ax.set_ylabel('Final workflow fraction (%)'); tidy(ax); panel(ax,'c','Contract tolerance changes finality'); ax.legend(loc='lower right',fontsize=5.8)
ax=axs[1,1]; mechs=['MCAR','BLOCK','CONTEXT_BIASED']; ml={'MCAR':'MCAR','BLOCK':'Block','CONTEXT_BIASED':'Context'}; mkmap={'MCAR':'o','BLOCK':'s','CONTEXT_BIASED':'^'}
positions=[]; labs=[]
for j,p in enumerate(['claude','codex']):
    base=j*3.65
    for k,m in enumerate(mechs): positions.append(base+k); labs.append(f'{p.title()}\n{ml[m]}')
# Left axis: RF prediction error; right axis: evidence-bound width. This avoids visually crushing the 2--3% error below the 62--66% interval width.
axr=ax.twinx()
for j,p in enumerate(['claude','codex']):
    base=j*3.65; xs=[]; ys=[]
    for k,m in enumerate(mechs):
        z=rg[(rg.dataset=='SyFI')&(rg.q==.2)&(rg.provider==p)&(rg.mechanism==m)&(rg.method=='random_forest_imputation')&(rg.metric=='median_relative_error')]
        if len(z): xs.append(base+k); ys.append(float(z.value.iloc[0])*100); ax.scatter(base+k,ys[-1],s=22,color=NAVY,marker=mkmap[m],edgecolor='white',linewidth=.35,zorder=3)
    ax.plot(xs,ys,color=NAVY,lw=.8,alpha=.8)
    xs=[]; ys=[]
    for k,m in enumerate(mechs):
        z=rg[(rg.dataset=='SyFI')&(rg.q==.2)&(rg.provider==p)&(rg.mechanism==m)&(rg.method=='EBMUS_evidence_bounds')&(rg.metric=='median_relative_width')]
        if len(z): xs.append(base+k); ys.append(float(z.value.iloc[0])*100); axr.scatter(base+k,ys[-1],s=22,color=TEAL,marker=mkmap[m],edgecolor='white',linewidth=.35,zorder=3)
    axr.plot(xs,ys,color=TEAL,lw=.8,alpha=.8)
ax.set_xticks(positions,labs,fontsize=5.8); ax.set_ylabel('RF MedRE (%)',color=NAVY); ax.set_ylim(1.8,2.8)
axr.set_ylabel('Evidence $W_{rel}$ (%)',color=TEAL); axr.set_ylim(59,69); axr.tick_params(axis='y',labelsize=6.5,colors=TEAL,direction='in')
tidy(ax); panel(ax,'d','TraceLab provider replication')
ax.legend(handles=[Line2D([0],[0],color=NAVY,marker='o',label='RF MedRE'),Line2D([0],[0],color=TEAL,marker='o',label='Evidence width')],loc='upper left',bbox_to_anchor=(.025,.975),fontsize=5.7,frameon=False,borderaxespad=0)
save(fig,'fig6_real_trace')

print('Generated quantitative Fig.3-Fig.6 and supplementary sensitivity figure.')
