"""Recompute public scalar arithmetic; never reconstruct private AUC/bootstrap inputs."""
from pathlib import Path
import hashlib,json,math
P=Path(__file__).resolve().parents[1]
SHA='f894ed3de4ad0a548d6076510b40754608f1d33ac305894d0bc3c9873b014281'
p=P/'evidence/aggregate_results.json'
if hashlib.sha256(p.read_bytes()).hexdigest()!=SHA:raise ValueError('Published scalar evidence changed')
x=json.loads(p.read_text())['MC4'];a=x['macro_report_proxy_auc']
arms={'affine_minus_plain':('mild_affine_bce','plain_bce'),'plain_minus_zero_update':('plain_bce','zero_update'),'affine_minus_zero_update':('mild_affine_bce','zero_update')}
result={}
for name,(left,right) in arms.items():
 item=x['comparisons'][name];delta=a[left]-a[right];values=item['per_finding_deltas']
 if len(values)!=x['findings'] or not all(math.isfinite(v) for v in values):raise ValueError('Invalid scalar finding deltas')
 mean=math.fsum(values)/len(values)
 if not math.isclose(delta,item['delta'],rel_tol=0,abs_tol=2e-15) or not math.isclose(mean,delta,rel_tol=0,abs_tol=2e-15):raise ValueError('Macro arithmetic does not match')
 low,high=item['descriptive_95pct_interval']
 if not all(math.isfinite(v) for v in (low,high)) or low>high:raise ValueError('Invalid descriptive interval')
 result[name]=dict(recomputed_macro_difference=delta,mean_of_twelve_finding_differences=mean,descriptive_95pct_interval=[low,high],includes_zero=low<=0<=high)
print(json.dumps(dict(status='PASS_PUBLISHED_SCALAR_ARITHMETIC',comparisons=result,decision=x['decision'],
 original_AUC_and_bootstrap_recomputed=False,study_level_inputs_read=False,official_score=None),indent=2,allow_nan=False))
