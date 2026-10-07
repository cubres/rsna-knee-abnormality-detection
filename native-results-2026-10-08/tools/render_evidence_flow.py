"""Original evidence-flow illustration using public scalar results only."""
from pathlib import Path
import argparse,hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
P=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-directory',required=True,type=Path);a=p.parse_args()
a.output_directory.mkdir(parents=True,exist_ok=False)
e=P/'evidence/aggregate_results.json'
if hashlib.sha256(e.read_bytes()).hexdigest()!='f894ed3de4ad0a548d6076510b40754608f1d33ac305894d0bc3c9873b014281':raise ValueError('Public scalar data changed')
x=json.loads(e.read_text());c=x['native_canary'];m=x['MC4']
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(16,5.7));fig.patch.set_facecolor('#f7f9fc');ax.set_facecolor('#f7f9fc');ax.set_xlim(0,14);ax.set_ylim(0,5.5);ax.axis('off')
ax.text(.25,5.05,'Two questions, two evidence paths',fontsize=19,weight='bold',color='#172840')
rows=[(3.2,'#2378ab','Mirror reader V12: does this implementation run faithfully?',[
 ('Three visible studies','Exact published reader\nFrozen publisher checkpoint'),('Native reader','Anatomical mirror\nbefore normalization'),('Reference checks','Volume/mask bytes match\nTwelve label rankings match'),('Narrow runtime PASS',f"{c['whole_native_elapsed_seconds']:.2f} seconds total\nNo own official score")]),
 (1.05,'#d58420','MC4 V59: does further fitting improve the starting point?',[
 ('Matched initialization','Two networks per arm\nSame starting-state hashes'),('Two fixed fine-tunes',f"Plain vs mild affine BCE\n4 epochs · {m['actual_AdamW_updates_per_network']:,} updates/network"),('Three sealed predictions','Unchanged, plain, affine\nBefore opening held targets'),('Retain unchanged','862 exposed proxy studies\nNeither final fine-tune promoted')])]
for y,col,title,cards in rows:
 ax.text(.25,y+1.37,title,fontsize=12,weight='bold',color=col)
 for i,(head,body) in enumerate(cards):
  xx=.25+3.45*i
  ax.add_patch(FancyBboxPatch((xx,y),2.9,1.08,boxstyle='round,pad=.12,rounding_size=.09',facecolor='white',edgecolor=col,linewidth=1.5))
  ax.text(xx+1.45,y+.8,head,ha='center',va='center',fontsize=11,weight='bold',color='#172840')
  ax.text(xx+1.45,y+.35,body,ha='center',va='center',fontsize=10,color='#52627b',linespacing=1.5)
  if i<3:ax.add_patch(FancyArrowPatch((xx+3.04,y+.54),(xx+3.29,y+.54),arrowstyle='-|>',mutation_scale=15,color=col,linewidth=1.5))
ax.text(.25,.28,'Canary: three visible studies only. MC4: previously exposed report proxies; patient independence unproven. No official-score claim.',fontsize=10,color='#52627b')
for suffix in ['svg','png']:
 fig.savefig(a.output_directory/('evidence_flow.'+suffix),dpi=160,bbox_inches='tight',facecolor=fig.get_facecolor(),metadata={'Title':'Knee evidence paths','Description':'Original explanatory diagram from public scalar results.'})
print(a.output_directory/'evidence_flow.png')
