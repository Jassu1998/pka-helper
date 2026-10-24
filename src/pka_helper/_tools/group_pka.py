#!/usr/bin/env python3
"""Macroscopic pKa of clusters of coupled acids (Asp/Glu) from a DelPhiPKa titration table.

   python3 tools/group_pka.py RUN_DIR [--cutoff 5.0] [--out clusters_macro.csv]
   python3 tools/group_pka.py titra.csv GLU0194A,GLU0204A            # explicit group

Why: acids that sit within a few A of each other share protons, so their single-site pKas split (one very high, one very low) and neither
is what an experiment sees when the group is titrated as a whole (e.g. the bacteriorhodopsin proton-release group Glu194/Glu204).
Clusters = connected sets of titrated Asp/Glu whose carboxylate centroids are closer than --cutoff A (default 5).
For a cluster of n sites the table holds the ionised (deprotonated) fraction f_i(pH); N(pH) = sum f_i rises from 0 to n and
  pKa_k  = pH where N crosses k-0.5   (macroscopic pKa of the k-th proton lost, k = 1..n)
  mid    = pH where N = n/2
Compare an experiment that follows the LAST proton released (shared-proton groups) with pKa_n, one that follows the whole group with mid."""
import argparse,csv,os,sys,numpy as np
def load_titra(p):
    L=[l.split() for l in open(p) if l.strip()];ph=np.array([float(x) for x in L[0][1:]])
    return ph,{r[0]:np.array([float(x) for x in r[1:]]) for r in L[1:]}
def cross(ph,N,level):
    """pH where N crosses `level`; +/-inf when it never does (saturated: protonated over the whole pH range -> above, ionised -> below)"""
    if N.max()<level: return float('inf')
    if N.min()>level: return float('-inf')
    for i in range(len(ph)-1):
        if (N[i]-level)*(N[i+1]-level)<=0 and N[i]!=N[i+1]:
            return float(ph[i]+(level-N[i])/(N[i+1]-N[i])*(ph[i+1]-ph[i]))
    return float('nan')
def centroids(pqr):
    ox={}
    for l in open(pqr):
        if l.startswith(('ATOM','HETATM')) and l[17:20] in('ASP','GLU') and l[12:16].strip() in('OD1','OD2','OE1','OE2'):
            ox.setdefault((l[17:20],l[21],int(l[22:26])),[]).append([float(l[30:38]),float(l[38:46]),float(l[46:54])])
    return {'%s%04d%s'%(k[0],k[2],k[1]):np.mean(v,0) for k,v in ox.items()}
def clusters(cent,ids,cutoff):
    ids=[i for i in ids if i in cent];par={i:i for i in ids}
    def f(x):
        while par[x]!=x:par[x]=par[par[x]];x=par[x]
        return x
    for a in range(len(ids)):
        for b in range(a+1,len(ids)):
            if np.linalg.norm(cent[ids[a]]-cent[ids[b]])<cutoff:par[f(ids[a])]=f(ids[b])
    g={}
    for i in ids:g.setdefault(f(i),[]).append(i)
    return [sorted(v) for v in g.values() if len(v)>1]
def fmt(v,lo=0.0,hi=14.0):
    return '>%g'%hi if v==float('inf') else '<%g'%lo if v==float('-inf') else '%.2f'%v
def report(ph,T,grp):
    N=sum(T[i] for i in grp);n=len(grp)
    return dict(members=' '.join(grp),n=n,single=' '.join('%s=%s'%(i,fmt(cross(ph,T[i],0.5))) for i in grp),
                macro=' '.join('pKa%d=%s'%(k,fmt(cross(ph,N,k-0.5))) for k in range(1,n+1)),mid=cross(ph,N,n/2.0),last=cross(ph,N,n-0.5))
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('target');ap.add_argument('groups',nargs='*');ap.add_argument('--cutoff',type=float,default=5.0);ap.add_argument('--out',default=None);ap.add_argument('--pqr',default=None)
    a=ap.parse_args(argv)
    if os.path.isdir(a.target):
        d=a.target;titra=os.path.join(d,'titra.csv');pqr=a.pqr or next((os.path.join(d,f) for f in ('protein_noHET.pqr',) if os.path.exists(os.path.join(d,f))),None)
        ph,T=load_titra(titra);ids=[i for i in T if i[:3] in('ASP','GLU')]
        if pqr is None:sys.exit('no protein_noHET.pqr in '+d)
        grps=clusters(centroids(pqr),ids,a.cutoff)
    else:
        ph,T=load_titra(a.target);grps=[g.split(',') for g in a.groups]
    rows=[report(ph,T,g) for g in sorted(grps)]
    out=a.out or (os.path.join(a.target,'clusters_macro.csv') if os.path.isdir(a.target) else None)
    if out:
        with open(out,'w',newline='') as fo:
            w=csv.DictWriter(fo,fieldnames=['members','n','single','macro','mid','last']);w.writeheader();[w.writerow(r) for r in rows]
    for r in rows:print('%-34s single: %-44s macro: %s  mid=%s'%(r['members'],r['single'],r['macro'],fmt(r['mid'])))
    print('[group_pka] %d cluster(s) of coupled acids (cutoff %.1f A)%s'%(len(rows),a.cutoff,'; written to '+out if out else ''))
# ---- all titratable groups (pka-helper results/clusters.txt) -------------------------------------------------------------------------
KEY={'ASP':('OD1','OD2'),'GLU':('OE1','OE2'),'HIS':('ND1','NE2'),'LYS':('NZ',),'ARG':('NE','NH1','NH2')}
def key_centroids(pqr):
    """{'HIS0027A': xyz} for every Asp/Glu/His/Lys/Arg (titrating atoms' centroid)."""
    ox={}
    for l in open(pqr):
        if l.startswith(('ATOM','HETATM')) and l[17:20] in KEY and l[12:16].strip() in KEY[l[17:20]]:
            ox.setdefault('%s%04d%s'%(l[17:20],int(l[22:26]),l[21]),[]).append([float(l[30:38]),float(l[38:46]),float(l[46:54])])
    return {k:np.mean(v,0) for k,v in ox.items()}
def protons(ph,T,grp):
    """protons bound by the group as a function of pH: acids (1 - ionised fraction) + bases (charged = protonated fraction)."""
    return sum((1.0-T[i]) if i[:3] in('ASP','GLU') else T[i] for i in grp)
def macro_general(ph,T,grp):
    """-> list of macroscopic pKas, highest first: pKa_k = pH where the bound-proton count falls through n - k + 0.5 (k-th proton lost)."""
    H=protons(ph,T,grp);n=len(grp)
    lost=[cross(ph,n-H,k-0.5) for k in range(1,n+1)]       # k-th proton lost as the pH rises (lowest pH first)
    return lost[::-1]                                        # highest first = the order protons are taken up as the pH falls
def all_clusters(run_dir,cutoff=5.0,ring_cutoff=12.0):
    """Coupled sets: key-atom centroids closer than `cutoff`, plus rings of the same residue in several chains (pore rings of oligomers)
    whose members are within `ring_cutoff` of each other.  -> list of member lists (DelPhiPKa names, e.g. HIS0027A)."""
    ph,T=load_titra(os.path.join(run_dir,'titra.csv'));cent=key_centroids(os.path.join(run_dir,'protein_noHET.pqr'))
    ids=[i for i in T if i in cent];par={i:i for i in ids}
    def f(x):
        while par[x]!=x:par[x]=par[par[x]];x=par[x]
        return x
    for a in range(len(ids)):
        for b in range(a+1,len(ids)):
            d=np.linalg.norm(cent[ids[a]]-cent[ids[b]]);same=ids[a][:7]==ids[b][:7]
            if d<cutoff or (same and d<ring_cutoff):par[f(ids[a])]=f(ids[b])
    g={}
    for i in ids:g.setdefault(f(i),[]).append(i)
    return ph,T,[sorted(v) for v in g.values() if len(v)>1]
def render_clusters(run_dir,cutoff=5.0):
    ph,T,grps=all_clusters(run_dir,cutoff)
    out=['Coupled groups (DelPhi titration): sites closer than %g A (titrating atoms), or the same residue in several chains within 12 A.'%cutoff,
         'An experiment that titrates such a group sees the MACROSCOPIC steps (protons lost one by one, highest first), not the per-residue pKas.',
         'These come from DelPhi alone (no PROPKA blend); compare them with stepwise measurements (e.g. the four pKas of a His tetrad).','']
    for grp in sorted(grps,key=lambda g:(-len(g),g)):
        single=' '.join('%s=%s'%(i,fmt(cross(ph,(1.0-T[i]) if i[:3] in('ASP','GLU') else 1.0-T[i],0.5))) for i in grp)
        steps=macro_general(ph,T,grp)
        out.append('%s\n    per residue: %s\n    macroscopic: %s   (mean %s)'%(' '.join(grp),single,' '.join(fmt(v) for v in steps),
                   fmt(float(np.mean([min(max(v,0),14) for v in steps])))))
    if not grps: out.append('none')
    return '\n'.join(out)
if __name__=='__main__':main()
