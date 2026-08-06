from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'figures'
DATA=ROOT/'data'
SUP=ROOT/'source_support'
FIG.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family':'serif','mathtext.fontset':'stix','font.size':8.2,
    'axes.labelsize':8.3,'axes.titlesize':8.8,'xtick.labelsize':7.2,
    'ytick.labelsize':7.2,'legend.fontsize':6.7,'figure.dpi':160,
    'savefig.dpi':400,'axes.linewidth':0.7,'lines.linewidth':1.25,
    'pdf.fonttype':42,'ps.fonttype':42,
})

CASES=[('Kn0p01',0.01,r'$Kn_D=0.01$'),('Kn0p025',0.025,r'$Kn_D=0.025$'),('Kn0p050',0.05,r'$Kn_D=0.05$')]

def save(fig,name):
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(FIG/f'{name}.png',bbox_inches='tight',dpi=400)
    plt.close(fig)

def panel(ax,s):
    ax.text(.015,.985,s,transform=ax.transAxes,ha='left',va='top',fontweight='bold',fontsize=8.8,
            bbox=dict(facecolor='white',edgecolor='none',alpha=.75,pad=1.2))

def load_cov(case):
    z=np.load(SUP/'covariance'/case/'inferred_covariances.npz')
    theta=z['theta_deg']; Cp=z['C_physical']; mode=z['physical_mode1'].astype(float)
    if np.mean(mode)<0: mode=-mode
    return theta,Cp,mode

def orthonormal_cosine_basis(theta,nmax=8):
    x=(theta-theta.min())/(theta.max()-theta.min())
    basis=[]
    for n in range(nmax):
        b=np.cos(n*np.pi*x)
        for q in basis: b=b-np.dot(b,q)*q
        b=b/np.linalg.norm(b)
        basis.append(b)
    return np.asarray(basis)

def corr_by_sep(theta,C,bin_width=1.0):
    d=np.sqrt(np.maximum(np.diag(C),0))
    R=np.clip(C/np.maximum(d[:,None]*d[None,:],1e-300),-1,1)
    sep=np.abs(theta[:,None]-theta[None,:])
    edges=np.arange(0,theta.max()-theta.min()+bin_width,bin_width)
    xc=[]; y=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(sep>=lo)&(sep<hi)&np.isfinite(R)
        if np.any(m):
            xc.append(.5*(lo+hi)); y.append(np.mean(R[m]))
    return np.asarray(xc),np.asarray(y)

# Derived angular data
mode_rows=[]; corr_rows=[]; physics=[]
for case,kn,label in CASES:
    theta,Cp,mode=load_cov(case)
    B=orthonormal_cosine_basis(theta,8)
    coeff=B@mode
    energy=coeff**2/np.dot(mode,mode)
    fit2=coeff[0]*B[0]+coeff[1]*B[1]
    xcorr,ycorr=corr_by_sep(theta,Cp)
    ev=np.linalg.eigvalsh(Cp); e1=ev[-1]/np.sum(np.maximum(ev,0))
    for j,(th,m,f) in enumerate(zip(theta,mode,fit2)):
        mode_rows.append({'case':case,'Kn':kn,'theta_deg':th,'mode1':m,'two_term_fit':f})
    for n,(c,e,ce) in enumerate(zip(coeff,energy,np.cumsum(energy))):
        physics.append({'case':case,'Kn':kn,'metric':f'cosine_coeff_{n}','value':c})
        physics.append({'case':case,'Kn':kn,'metric':f'cosine_energy_{n}','value':e})
        physics.append({'case':case,'Kn':kn,'metric':f'cosine_cumulative_{n+1}','value':ce})
    for s,r in zip(xcorr,ycorr): corr_rows.append({'case':case,'Kn':kn,'separation_deg':s,'correlation':r})
    physics.append({'case':case,'Kn':kn,'metric':'physical_covariance_E1','value':e1})

pd.DataFrame(mode_rows).to_csv(DATA/'angular_mode_two_term_decomposition.csv',index=False)
pd.DataFrame(corr_rows).to_csv(DATA/'angular_correlation_by_separation.csv',index=False)

# Inference, geometry and full-field statistics
common=pd.read_csv(DATA/'common200_correlated_noise_summary.csv')
full=pd.read_csv(DATA/'full_record_correlated_noise_summary.csv')
pod=pd.read_csv(DATA/'corrected_pod_summary.csv')
geom=pod.groupby('Kn')[['s_marker_over_R','delta_over_R']].first().reset_index()
cons=pd.read_csv(DATA/'multimoment_consensus.csv')
metrics=pd.read_csv(DATA/'displacement_template_metrics.csv')

def inference_row(case):
    src=common if case=='Kn0p01' else full
    return src[(src.case==case)&(src.angular_smoothing_rays==1)].iloc[0]

phys_rows=[]
for case,kn,label in CASES:
    r=inference_row(case); g=geom[np.isclose(geom.Kn,kn)].iloc[0]
    delta=float(g.delta_over_R); amp=float(r.global_physical_std_R); tau=float(r.tau_physical_exponential_star)
    pm=[p for p in physics if p['case']==case]
    get=lambda key: next(x['value'] for x in pm if x['metric']==key)
    phys_rows.append({
        'case':case,'Kn':kn,'delta_over_R':delta,'global_displacement_std_over_R':amp,
        'displacement_std_over_delta':amp/delta,'tau_star':tau,
        'local_layer_crossing_star':delta/2,'tau_over_local_crossing':tau/(delta/2),
        'physical_covariance_E1':get('physical_covariance_E1'),
        'uniform_basis_energy':get('cosine_energy_0'),
        'two_term_basis_energy':get('cosine_cumulative_2'),
        'uniform_mode_correlation':float(r.uniform_mode_correlation),
        'far_angle_mean_correlation':float(r.far_angle_mean_correlation),
        'tau_q025':float(r.bootstrap_tau_physical_exponential_star_q025),
        'tau_q975':float(r.bootstrap_tau_physical_exponential_star_q975),
    })
phys=pd.DataFrame(phys_rows)
phys.to_csv(DATA/'dynamic_physics_summary.csv',index=False)
pd.DataFrame(physics).to_csv(DATA/'angular_basis_metrics_long.csv',index=False)

# ------------------------------------------------------------------
# Fig 11: angular kinematics and correlation range
# ------------------------------------------------------------------
mode_df=pd.DataFrame(mode_rows); corr_df=pd.DataFrame(corr_rows)
fig,axes=plt.subplots(2,2,figsize=(7.25,5.0),constrained_layout=True)
ax=axes[0,0]
for case,kn,label in CASES:
    g=mode_df[mode_df.case==case]
    ax.plot(g.theta_deg,g.mode1,label=label,marker='o',markevery=9,ms=2.5)
ax.set_xlabel(r'$\theta$ (deg)'); ax.set_ylabel(r'$g_1(\theta)$')
ax.set_title('Inferred physical angular mode'); ax.grid(alpha=.22); ax.legend(); panel(ax,'(a)')

ax=axes[0,1]
for case,kn,label in CASES[:2]:
    g=mode_df[mode_df.case==case]
    ax.plot(g.theta_deg,g.mode1,label=label+' inferred')
    ax.plot(g.theta_deg,g.two_term_fit,ls='--',label=label+' 2-term')
ax.set_xlabel(r'$\theta$ (deg)'); ax.set_ylabel('mode amplitude')
ax.set_title('Translation plus curvature modulation'); ax.grid(alpha=.22); ax.legend(ncol=2,fontsize=6.0); panel(ax,'(b)')

ax=axes[1,0]
for case,kn,label in CASES:
    v=[next(x['value'] for x in physics if x['case']==case and x['metric']==f'cosine_cumulative_{n}') for n in range(1,7)]
    ax.plot(range(1,7),v,marker='o',label=label)
ax.axhline(.95,ls='--',color='black',lw=.8)
ax.set_xlabel('number of angular basis functions'); ax.set_ylabel('cumulative mode-shape energy')
ax.set_ylim(.5,1.01); ax.set_title('Low-order angular representation'); ax.grid(alpha=.22); ax.legend(); panel(ax,'(c)')

ax=axes[1,1]
for case,kn,label in CASES:
    g=corr_df[corr_df.case==case]
    ax.plot(g.separation_deg,g.correlation,label=label)
ax.axhline(0,color='black',lw=.75)
ax.axvline(15,color='black',lw=.75,ls=':')
ax.set_xlabel(r'angular separation $|\Delta\theta|$ (deg)'); ax.set_ylabel('physical correlation')
ax.set_title('Correlation across the forebody sector'); ax.grid(alpha=.22); ax.legend(); panel(ax,'(d)')
fig.suptitle('Angular kinematics of the collective displacement coordinate',fontsize=10.5)
save(fig,'fig11_angular_kinematics')

# ------------------------------------------------------------------
# Fig 12: amplitude, memory and high-rank/low-rank separation
# ------------------------------------------------------------------
fig,axes=plt.subplots(2,2,figsize=(7.25,4.9),constrained_layout=True)
ax=axes[0,0]
ax.semilogx(phys.Kn,100*phys.displacement_std_over_delta,marker='o')
for _,r in phys.iterrows(): ax.annotate(f"{100*r.displacement_std_over_delta:.2f}%",(r.Kn,100*r.displacement_std_over_delta),xytext=(3,4),textcoords='offset points',fontsize=6.5)
ax.set_xlabel(r'$Kn_D$'); ax.set_ylabel(r'$\sigma_a/\delta_{10-90}$ (\%)')
ax.set_title('Linear-small displacement amplitude'); ax.grid(alpha=.22); panel(ax,'(a)')

ax=axes[0,1]
y=phys.tau_star.to_numpy(); lo=phys.tau_q025.to_numpy(); hi=phys.tau_q975.to_numpy()
ax.errorbar(phys.Kn,y,yerr=[np.maximum(y-lo,0),np.maximum(hi-y,0)],marker='o',capsize=3,label=r'$\tau_p^*$')
ax.semilogx(phys.Kn,phys.local_layer_crossing_star,marker='s',label=r'$\delta_{10-90}/D$')
ax.set_xlabel(r'$Kn_D$'); ax.set_ylabel('convective time')
ax.set_title('Body-scale memory versus local crossing time'); ax.grid(alpha=.22); ax.legend(); panel(ax,'(b)')

ax=axes[1,0]
combined=pod[pod.run=='common200_multivariate'].set_index('Kn')
fieldE=np.array([combined.loc[k,'E1'] for k in phys.Kn])
ax.semilogx(phys.Kn,100*phys.physical_covariance_E1,marker='o',label='front physical covariance')
ax.semilogx(phys.Kn,100*fieldE,marker='s',label='complete-field POD')
ax.set_xlabel(r'$Kn_D$'); ax.set_ylabel('leading fraction (\%)')
ax.set_title('Low-dimensional interface inside a high-rank field'); ax.grid(alpha=.22); ax.legend(); panel(ax,'(c)')

ax=axes[1,1]
ax.semilogx(cons.Kn,cons.field_pc1_variance_fraction,marker='o',label='multi-moment PC1 fraction')
ax.semilogx(cons.Kn,cons.field_pc1_marker_correlation,marker='s',label='PC1-marker correlation')
ax.semilogx(cons.Kn,cons.median_cross_variable_correlation,marker='^',label='median cross-moment correlation')
ax.set_xlabel(r'$Kn_D$'); ax.set_ylabel('fraction or correlation'); ax.set_ylim(0,1)
ax.set_title('Slaving of macroscopic moments'); ax.grid(alpha=.22); ax.legend(); panel(ax,'(d)')
fig.suptitle('Amplitude, memory and multi-moment organization',fontsize=10.5)
save(fig,'fig12_body_scale_memory')

# ------------------------------------------------------------------
# Fig 13: spatial localization and multi-field template evidence
# ------------------------------------------------------------------
shift=[]
for case,kn,label in CASES:
    d=pd.read_csv(SUP/'template'/case/'template_shift_null.csv')
    shift.append(d)
shift=pd.concat(shift,ignore_index=True)
shift.to_csv(DATA/'template_shift_null_all.csv',index=False)

fig,axes=plt.subplots(2,2,figsize=(7.25,4.9),constrained_layout=True)
for ax,var,title,lab in [(axes[0,0],'D','Density translation template','(a)'),(axes[0,1],'P','Pressure translation template','(b)')]:
    for case,kn,label in CASES:
        g=shift[(shift.case==case)&(shift.variable==var)].sort_values('xi_shift')
        ax.plot(g.xi_shift,100*g.median_projection_fraction,marker='o',label=label)
    ax.axvline(0,color='black',ls=':',lw=.8)
    ax.set_xlabel(r'template shift $\Delta\xi$'); ax.set_ylabel('median projected variance (\%)')
    ax.set_title(title); ax.grid(alpha=.22); ax.legend(); panel(ax,lab)

ax=axes[1,0]
for var,label in [('D',r'$\rho$'),('P',r'$p$'),('MA',r'$M$'),('TTR',r'$T_{tr}$')]:
    g=metrics[metrics.variable==var].sort_values('Kn')
    y=g.field_marker_corr.to_numpy(); lo=g.field_marker_corr_q025.to_numpy(); hi=g.field_marker_corr_q975.to_numpy()
    ax.errorbar(g.Kn,y,yerr=[y-lo,hi-y],marker='o',capsize=2,label=label)
ax.axhline(0,color='black',lw=.75)
ax.set_xlabel(r'$Kn_D$'); ax.set_ylabel('full-field/marker correlation')
ax.set_title('Independent field recovery of displacement'); ax.grid(alpha=.22); ax.legend(); panel(ax,'(c)')

ax=axes[1,1]
for var,label in [('D',r'$\rho$'),('P',r'$p$'),('MA',r'$M$'),('TTR',r'$T_{tr}$')]:
    g=metrics[metrics.variable==var].sort_values('Kn')
    ax.semilogx(g.Kn,100*g.median_projection_fraction,marker='o',label=label)
ax.set_xlabel(r'$Kn_D$'); ax.set_ylabel('median projected variance (\%)')
ax.set_title('Weak coordinate in broadband fluctuations'); ax.grid(alpha=.22); ax.legend(); panel(ax,'(d)')
fig.suptitle('Spatial localization and full-field validation of translation',fontsize=10.5)
save(fig,'fig13_template_localization')

print(phys.to_string(index=False))
