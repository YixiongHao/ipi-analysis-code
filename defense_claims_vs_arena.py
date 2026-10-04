#!/usr/bin/env python3
"""Claimed-vs-arena defense comparison: two stacked tables (system-level, then
classifier/detector), each with a verdict badge.

System-level defenses are labelled with the design pattern used in the paper's
taxonomy (Table: defenses we evaluate), following Beurer-Kellner et al.,
"Design Patterns for Securing LLM Agents against Prompt Injections",
arXiv:2506.08837 (Action-Selector / Plan-Then-Execute / Map-Reduce / Dual-LLM /
Code-Then-Execute / Context-Minimization). MELON and CausalArmor sit outside the
six patterns; they detect at runtime rather than restructure the agent loop.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

# ---- palette (light lab-report) ----
INK="#191e23"; MUTED="#5b656e"; FAINT="#8b949c"
HAIR="#dfe4e8"; HDRBG="#eef1f3"; GROUND="#ffffff"; SURF="#ffffff"; ACC="#0e6f73"
STRIPE="#f7f9fa"
SEM={"match":("#1f7a4d","#e4f2ea"),"partial":("#a56a06","#f6ecd8"),
     "miss":("#b23838","#f6e2e2"),"na":("#63707a","#e8ebed")}
GLYPH={"match":"✓","partial":"≈","miss":"✗","na":"–"}
MONO={"family":"monospace"}; SANS={"family":"DejaVu Sans"}

# ---------------- data ----------------
# system: name, pattern, approx?, paper_reduction, paper_sub, arena_pct, arena_sub, verdict, sem
SYS=[
 ("IPIGuard","Plan-Then-Execute",False,"94.8%","13.16 → 0.69%  ADojo","97.3%","61.9 → 1.6%","Holds","match"),
 ("FIDES","Dual-LLM (via IFC)",False,"~100%","→ 0%  ADojo (IFC)","93.6%","61.9 → 4.0%","Holds","match"),
 ("CaMeL","Code-Then-Execute",False,"~100%","0% in-model  ADojo","91.2%","61.9 → 5.5%","Holds","match"),
 ("Firewalls","Context-Minimization",False,"~100%","0–0.02%  ADojo","72.5%","61.9 → 17.0%","Falls short","miss"),
 ("MELON","Runtime detection",False,"98.5%","16 → 0.24%  ADojo","70.8%","61.9 → 18.0%","Falls short","miss"),
 ("CausalArmor","Runtime detection",False,"95.9%","88.9 → 3.65% DoomArena","62.9%","61.9 → 23.0%","Falls short","miss"),
]
# classifier: name, sub, claim_main, claim_sub, recall, fpr, verdict, sem
CLS=[
 ("Cygnal","commercial, hosted API","N/A","","80.0%","FPR 0.0 / 0.2%","No claim","na"),
 ("StackOne Defender T2","MiniLM multi-head","89% balanced accuracy","vendor claim · not recall","76.5%","FPR 5.4 / 10.1%","Loose","partial"),
 ("ProtectAI v2","DeBERTa-v3-base","recall 99.7% · F1 95.5%","20k held-out","57.5%","FPR 23.3 / 73.2%","Large gap","miss"),
 ("PromptGuard-2 86M","Meta","recall 97.5% @1% FPR","ADojo prevent 81.2%","27.9%","FPR 0.3 / 3.1%","Large gap","miss"),
 ("DataSentinel","Mistral-7B, canary","TPR ~99% · FPR ~0","OpenPromptInjection","23.6%","FPR 8.8 / 8.7%","Large gap","miss"),
]

# ---------------- figure ----------------
# Content runs from y=0.965 down to y≈-0.069 (legend bottom): the ylim must
# cover it or patches clip at the axes edge (text doesn't, so rows half-render).
# FH is scaled by the same 1.085 span so row heights stay the same in inches.
FW,FH=9.8,9.7
fig=plt.figure(figsize=(FW,FH),dpi=200); fig.patch.set_facecolor(GROUND)
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,1); ax.set_ylim(-0.085,1); ax.axis("off")
ASPECT=FW/FH   # for width-correct rounded badges

X0,X1=0.030,0.970
W=X1-X0

def badge(cx_right,ycen,text,sem):
    """Right-anchored verdict pill: glyph + text on a rounded, tinted rectangle."""
    label=f"{GLYPH[sem]}  {text}"
    fg,bg=SEM[sem]
    # width heuristic in axis units (monospace-ish); height fixed
    w=0.012*len(label)+0.020
    h=0.040
    x0=cx_right-w
    ax.add_patch(FancyBboxPatch((x0,ycen-h/2),w,h,
                 boxstyle="round,pad=0.002,rounding_size=0.014",
                 mutation_aspect=ASPECT,fc=bg,ec=fg,lw=0.9,zorder=4))
    ax.text(x0+w/2,ycen,label,color=fg,fontsize=10.5,fontweight="bold",
            ha="center",va="center",zorder=5,**SANS)

def panel(ytop,title,cols,rows,cellfn,rh):
    edges=[]; cx=X0
    for _,w,_ in cols:
        edges.append((cx,cx+w*W)); cx+=w*W
    ax.text(X0,ytop,title,color=INK,fontsize=15,fontweight="bold",va="top",**SANS)
    y=ytop-0.040
    hh=0.032
    ax.add_patch(Rectangle((X0,y-hh),W,hh,facecolor=HDRBG,edgecolor=HAIR,lw=0.7))
    for (name,w,al),(a,b) in zip(cols,edges):
        hx=a+0.006 if al=="left" else b-0.006
        ax.text(hx,y-hh/2,name,color=MUTED,fontsize=10,fontweight="bold",
                va="center",ha=("left" if al=="left" else "right"),**SANS)
    y-=hh
    top0=y
    for i,r in enumerate(rows):
        ax.add_patch(Rectangle((X0,y-rh),W,rh,facecolor=(STRIPE if i%2 else SURF),edgecolor="none"))
        cellfn(r,edges,y,rh)
        ax.plot([X0,X1],[y-rh,y-rh],color=HAIR,lw=0.6)
        y-=rh
    ax.add_patch(Rectangle((X0,y),W,top0-y,fill=False,edgecolor=HAIR,lw=1.0))
    return y

# ---- system panel ----
SYS_COLS=[("Defense  /  design pattern",0.30,"left"),
          ("Reported reduction",0.27,"left"),
          ("Arena reduction",0.24,"left"),
          ("Verdict",0.19,"right")]
def sys_cell(r,edges,ytop,h):
    name,pat,approx,pred,psub,ared,asub,verdict,sem=r
    stext=SEM[sem][0]
    x=edges[0][0]+0.008
    ax.text(x,ytop-h*0.32,name,color=INK,fontsize=12.5,fontweight="bold",va="center",**SANS)
    ax.text(x,ytop-h*0.72,pat+("  (approx.)" if approx else ""),color=ACC,fontsize=9.6,va="center",**SANS)
    x=edges[1][0]+0.008
    ax.text(x,ytop-h*0.34,pred,color=INK,fontsize=13.5,fontweight="bold",va="center",**MONO)
    ax.text(x,ytop-h*0.74,psub,color=MUTED,fontsize=9,va="center",**SANS)
    x=edges[2][0]+0.008
    ax.text(x,ytop-h*0.34,ared,color=stext,fontsize=13.5,fontweight="bold",va="center",**MONO)
    ax.text(x,ytop-h*0.74,asub,color=MUTED,fontsize=9,va="center",**MONO)
    badge(edges[3][1]-0.006,ytop-h*0.5,verdict,sem)

# ---- classifier panel ----
CLS_COLS=[("Detector",0.30,"left"),
          ("Paper / vendor claim",0.31,"left"),
          ("Arena recall",0.22,"left"),
          ("Verdict",0.17,"right")]
def cls_cell(r,edges,ytop,h):
    name,sub,cmain,csub,recall,fpr,verdict,sem=r
    stext=SEM[sem][0]
    x=edges[0][0]+0.008
    ax.text(x,ytop-h*0.32,name,color=INK,fontsize=12,fontweight="bold",va="center",**SANS)
    ax.text(x,ytop-h*0.72,sub,color=FAINT,fontsize=9,va="center",**SANS)
    x=edges[1][0]+0.008
    ax.text(x,ytop-(h*0.34 if csub else h*0.5),cmain,color=INK,fontsize=10.5,va="center",**SANS)
    if csub:
        ax.text(x,ytop-h*0.74,csub,color=MUTED,fontsize=9,va="center",**SANS)
    x=edges[2][0]+0.008
    ax.text(x,ytop-h*0.34,recall,color=stext,fontsize=14.5,fontweight="bold",va="center",**MONO)
    ax.text(x,ytop-h*0.76,fpr,color=MUTED,fontsize=8.8,va="center",**MONO)
    badge(edges[3][1]-0.006,ytop-h*0.5,verdict,sem)

ybot=panel(0.965,"System-level defenses — attack-success-rate reduction",
           SYS_COLS,SYS,sys_cell,rh=0.072)
ybot=panel(ybot-0.052,"Classifier / detector defenses — injection detection",
           CLS_COLS,CLS,cls_cell,rh=0.072)

# ---- footer legend ----
ly=ybot-0.030
lx=X0
for sem,lab in [("match","reproduces"),("partial","loose / caveated"),
                ("miss","gap on arena"),("na","no comparable claim")]:
    fg,bg=SEM[sem]
    ax.add_patch(FancyBboxPatch((lx,ly-0.014),0.026,0.028,
                 boxstyle="round,pad=0.002,rounding_size=0.010",
                 mutation_aspect=ASPECT,fc=bg,ec=fg,lw=0.9))
    ax.text(lx+0.013,ly,GLYPH[sem],color=fg,fontsize=9.5,fontweight="bold",
            ha="center",va="center",**SANS)
    ax.text(lx+0.034,ly,lab,color=MUTED,fontsize=9.3,va="center",**SANS)
    lx+=0.034+0.012*len(lab)+0.030

import os
out=os.path.join(os.path.dirname(os.path.abspath(__file__)),"defense_claims_vs_arena.png")
fig.savefig(out,dpi=200,facecolor=GROUND,bbox_inches="tight",pad_inches=0.04)
print("saved",out)
