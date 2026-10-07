"""Render scientific aggregate-only charts from actual qualified V59 receipt."""
from pathlib import Path
import argparse
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output-directory',required=True,type=Path)
a=p.parse_args()
a.output_directory.mkdir(parents=True,exist_ok=False)
RECEIPT=HERE/'evidence/aggregate_results.json'
SHA='f894ed3de4ad0a548d6076510b40754608f1d33ac305894d0bc3c9873b014281'
if hashlib.sha256(RECEIPT.read_bytes()).hexdigest()!=SHA:raise ValueError('Qualified aggregate receipt changed')
x=json.loads(RECEIPT.read_text())['MC4']
if x['status']!='PASS_NATIVE_EXECUTION_AND_EXPOSED_PROXY_AGGREGATE_QUALIFICATION':raise ValueError('No qualified result')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,(left,right)=plt.subplots(1,2,figsize=(15.5,4.9),gridspec_kw={'width_ratios':[.95,1.4]},layout='constrained')
fig.patch.set_facecolor('#f7f9fc')
for ax in (left,right):ax.set_facecolor('#f7f9fc');ax.spines['left'].set_color('#c1c9d8');ax.spines['bottom'].set_color('#c1c9d8')
labels=['Unchanged initialization','Plain BCE fine-tune','Mild affine fine-tune'];arms=['zero_update','plain_bce','mild_affine_bce'];colors=['#283a59','#2378ab','#d58420']
for y,(label,arm,col) in enumerate(zip(labels,arms,colors)):
 value=x['macro_report_proxy_auc'][arm];left.scatter(value,y,s=110,color=col,zorder=3)
 left.text(value+.00016,y,f'{value:.6f}',ha='left',va='center',fontweight='bold',color=col)
left.set_yticks(range(3),labels);left.invert_yaxis();left.set_ylim(2.65,-.65);left.set_xlim(.8761,.8807);left.grid(axis='x',color='#dbe1e9',linewidth=.7);left.set_xlabel('Macro ROC AUC against exposed report proxies');left.set_title('Unchanged initialization remained highest',loc='left',fontweight='bold',pad=15)
names=['affine_minus_plain','plain_minus_zero_update','affine_minus_zero_update'];display=['Mild affine − plain BCE','Plain BCE − unchanged','Mild affine − unchanged']
for y,(name,label) in enumerate(zip(names,display)):
 item=x['comparisons'][name];delta=item['delta'];low,high=item['descriptive_95pct_interval'];col='#d58420' if y==0 else '#2378ab'
 right.errorbar(delta,y,xerr=[[delta-low],[high-delta]],fmt='o',color=col,capsize=6,linewidth=2,markersize=7,zorder=4)
 right.text(.00220,y,f'{delta:+.6f}',ha='left',va='center',color=col,fontweight='bold')
right.axvline(0,color='#52627b',linestyle='--',linewidth=1);right.set_yticks(range(3),display);right.invert_yaxis();right.set_ylim(2.65,-.65);right.set_xlim(-.0047,.0039);right.set_xticks([-.004,-.002,0,.002]);right.grid(axis='x',color='#dbe1e9',linewidth=.7);right.set_xlabel('Paired macro AUC difference · pointwise descriptive 95% interval');right.set_title('Neither fine-tune beat the unchanged reference',loc='left',fontweight='bold',pad=15)
fig.suptitle('Knee MC4 V59: qualified execution, no promotion evidence',fontsize=17,fontweight='bold',color='#172840')
fig.text(.015,-.035,'862 exposed nonGold studies · 12 findings · 2,000 paired whole-study bootstrap draws · single training seed\nReport overlap unknown; patient independence unproven. These are local proxy metrics, not official Kaggle scores.',ha='left',va='top',fontsize=10,color='#52627b')
for suffix in ('svg','png'):
 path=a.output_directory/f'V59_EXPOSED_PROXY_COMPARISON.{suffix}'
 if path.exists():raise FileExistsError(path)
 fig.savefig(path,dpi=160,bbox_inches='tight',facecolor=fig.get_facecolor(),metadata={'Title':'Knee MC4 V59 exposed report-proxy comparison','Description':'Measured aggregate evidence; not official score. Input receipt SHA256 '+SHA} if suffix=='svg' else {'Title':'Knee MC4 V59 exposed report-proxy comparison','Description':'Measured aggregate evidence; not official score. Input receipt SHA256 '+SHA})
print(str(a.output_directory/'V59_EXPOSED_PROXY_COMPARISON.png'))
