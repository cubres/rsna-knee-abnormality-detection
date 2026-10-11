"""Invented, bounded CPU controls for the generic factor/index receipt joins."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import torch
import factor_join as fj


def main(output):
    began=time.process_time();out=Path(output);out.mkdir(exist_ok=False)
    torch.manual_seed(43);torch.set_num_threads(1)
    labels=((torch.arange(9).view(-1,1)+torch.arange(2))%2).float().contiguous()
    weight=torch.tensor([1.0,1.2]);ids=['invented-record-'+str(i) for i in range(9)]
    data_sha=fj.digest('invented-data-only')
    bindings=fj.input_bindings(labels,weight,ids,data_sha);registered=copy.deepcopy(bindings)
    cases=[]
    def control(name,callback,reject=True):
        try:result=callback()
        except fj.FactorHold as error:
            fj.require(reject,'unexpected rejection '+name)
            cases.append({'case':name,'status':'PASS_REJECTED','reason':str(error)})
        else:
            fj.require(not reject,'negative control admitted '+name)
            cases.append({'case':name,'status':'PASS_ACCEPTED'})
    def bind(ys=labels,pw=weight,uids=ids,dh=data_sha,reg=registered):
        return fj.validate_registered_inputs(bindings,reg,ys,pw,uids,dh)
    control('matching saved and registered factors',lambda:bind(),False)
    ys=labels.clone();ys[0,0]=1-ys[0,0]
    control('changed labels with fresh matching registration',lambda:bind(ys=ys,reg=fj.input_bindings(ys,weight,ids,data_sha)))
    pw=weight.clone();pw[0]+=0.1
    control('changed weights with fresh matching registration',lambda:bind(pw=pw,reg=fj.input_bindings(labels,pw,ids,data_sha)))
    shifted=list(ids);shifted[0],shifted[1]=shifted[1],shifted[0]
    control('changed ordered IDs with fresh matching registration',lambda:bind(uids=shifted,reg=fj.input_bindings(labels,weight,shifted,data_sha)))
    changed_sha=fj.digest('other invented data')
    control('changed data with fresh matching registration',lambda:bind(dh=changed_sha,reg=fj.input_bindings(labels,weight,ids,changed_sha)))
    control('missing saved bindings',lambda:fj.validate_registered_inputs(None,registered,labels,weight,ids,data_sha))
    control('missing registration',lambda:bind(reg=None))
    single=labels[:3,:1].clone().contiguous();one_weight=weight[:1].clone();one_ids=ids[:3]
    one_bindings=fj.input_bindings(single,one_weight,one_ids,data_sha)
    bool_shape=copy.deepcopy(one_bindings);bool_shape['labels_pin']['shape'][1]=True
    control('boolean dimension differs from integer dimension',lambda:fj.validate_registered_inputs(bool_shape,one_bindings,
                                                                                                  single,one_weight,one_ids,data_sha))
    control('malformed data pin',lambda:fj.input_bindings(labels,weight,ids,'missing'))
    duplicate=list(ids);duplicate[1]=duplicate[0]
    control('duplicate identity namespace',lambda:fj.input_bindings(labels,weight,duplicate,data_sha))
    noncontiguous=torch.zeros(9,4)[:,::2];noncontiguous.copy_(labels)
    control('noncontiguous labels',lambda:fj.input_bindings(noncontiguous,weight,ids,data_sha))
    nan=labels.clone();nan[0,0]=float('nan')
    control('nonfinite labels',lambda:fj.input_bindings(nan,weight,ids,data_sha))
    control('weight length mismatch',lambda:fj.input_bindings(labels,weight[:1],ids,data_sha))
    class Base:
        def __init__(self):self.uids=list(ids);self.calls=[]
        def __len__(self):return len(self.uids)
        def __getitem__(self,index):
            self.calls.append(index)
            return torch.arange(4).float()+index/10,labels[index],torch.tensor([0.0,1.0],dtype=torch.float64)
    base=Base();dataset=fj.IndexedStudyDataset(base,ids)
    batch=next(iter(torch.utils.data.DataLoader(dataset,batch_size=2,sampler=[4,2],num_workers=0)))
    x,y,metadata,observed=batch
    fj.require(base.calls==[4,2] and observed.tolist()==[4,2] and torch.equal(y,labels[[4,2]]),'actual loader index join failed')
    model=torch.nn.Linear(4,2);opt=torch.optim.AdamW(model.parameters(),lr=0.01)
    before=model.weight.detach().clone()
    loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x),y,pos_weight=weight)
    fj.require(bool(torch.isfinite(loss)),'invented CPU loss nonfinite');loss.backward();opt.step();opt.zero_grad(set_to_none=True)
    fj.require(not torch.equal(model.weight.detach(),before),'actual invented CPU parameter update absent')
    control('base identity order drift',lambda:fj.IndexedStudyDataset(base,shifted))
    base.uids[4]='changed-record'
    control('base identity drift at observed index',lambda:dataset[4])
    control('negative index',lambda:dataset[-1])
    control('boolean index',lambda:dataset[True])
    class BadBase(Base):
        def __getitem__(self,index):return [1,2,3]
    control('base reader must return exactly three values',lambda:fj.IndexedStudyDataset(BadBase(),ids)[0])
    receipt={'kind':'invented-evaluation','fixed_factor_sha256':fj.digest(bindings),'caller_declared_metric':0.4}
    path=out/'immutable.json';ref=fj.create_only_receipt(path,receipt);prior=path.read_bytes()
    control('retry preserves prior receipt',lambda:fj.create_only_receipt(path,receipt))
    fj.require(path.read_bytes()==prior and hashlib.sha256(prior).hexdigest()==ref['sha256'],'immutable receipt changed')
    partial=out/'partial.json'
    with partial.open('x') as stream:stream.write('{')
    control('partial bytes are not repaired',lambda:fj.create_only_receipt(partial,receipt))
    fj.require(partial.read_text()=='{','partial evidence removed')
    control('notebook metadata path rejected',lambda:fj.create_only_receipt(out/'kernel-metadata.json',receipt))
    record={'status':'PASS_GENERIC_FACTOR_JOINS','invented_fixture_only':True,'invented_records':9,'invented_findings':2,
        'actual_cpu_microbatches':1,'actual_observed_indices':observed.tolist(),'actual_cpu_parameter_change_observed':True,
        'controls':cases,'count':len(cases),'cpu_s':time.process_time()-began,'native_execution':False,
        'metric_recomputed':False,'module_sha256':hashlib.sha256(Path(fj.__file__).read_bytes()).hexdigest(),
        'tests_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'all_invalid_artifacts_preserved':True}
    with (out/'GENERIC_FACTOR_RECEIPT.json').open('x') as stream:stream.write(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    os.nice(5);resource.setrlimit(resource.RLIMIT_CPU,(15,15))
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);main(parser.parse_args().output)
