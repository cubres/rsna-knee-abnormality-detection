"""MIT original read-only gradient/scaler observations; no step or recovery.

The caller owns the declared phase, source and interpretation. This module
never fixes gradients, updates a scaler/optimizer or qualifies a training run.
Snapshots use scalar reductions rather than CPU copies of model tensors.
"""
import hashlib
import math
import time

import torch


def capture(model, scaler, phase, wall_budget_s=20., max_parameter_records=512):
    if phase not in('after_scaled_backward_before_unscale','after_unscale_before_clip'):
        raise ValueError('explicit registered diagnostic phase required')
    if type(wall_budget_s)not in(int,float)or not math.isfinite(wall_budget_s)or not 0<wall_budget_s<=30:
        raise ValueError('bounded diagnostic observation budget')
    if type(max_parameter_records)is not int or not 1<=max_parameter_records<=1024:
        raise ValueError('bounded parameter metadata record count')
    start=time.monotonic();scale=scaler.get_scale();state=scaler.state_dict()
    if type(scale)not in(int,float)or not math.isfinite(scale)or scale<=0:
        raise ValueError('actual scaler scale invalid')
    if type(state)is not dict:
        raise ValueError('actual scaler state dictionary required')
    # Public scalar state is retained verbatim after a narrow typed check.
    if not all(type(k)is str and type(v)in(int,float,bool,str,type(None))
               and(not isinstance(v,float)or math.isfinite(v))for k,v in state.items()):
        raise ValueError('unexpected scaler public state schema')
    records=[];total=missing=nonfinite=finite_count=nan_count=inf_count=elements=0
    dtype_counts={};complete=True;unsupported=0
    for name,parameter in model.named_parameters():
        if not parameter.requires_grad:continue
        if time.monotonic()-start>wall_budget_s:complete=False;break
        total+=1;gradient=parameter.grad
        record={'parameter_name_sha256':hashlib.sha256(name.encode()).hexdigest(),
                'parameter_shape':list(parameter.shape),'gradient_missing':gradient is None}
        if gradient is None:
            missing+=1;record['classification']='MISSING_GRADIENT'
        else:
            value=gradient.detach();record.update(gradient_dtype=str(value.dtype),gradient_shape=list(value.shape))
            dtype_counts[str(value.dtype)]=dtype_counts.get(str(value.dtype),0)+1
            elements+=value.numel()
            if value.is_sparse or not(value.is_floating_point()or value.is_complex()):
                unsupported+=1;record['classification']='UNREVIEWED_GRADIENT_LAYOUT_OR_DTYPE'
            else:
                is_finite=bool(torch.isfinite(value).all())
                if is_finite:
                    finite_count+=1;record['classification']='FINITE_PRESENT_GRADIENT'
                    record['maximum_absolute_gradient']=float(value.abs().max()) if value.numel() else 0.
                else:
                    nonfinite+=1;n=int(torch.isnan(value).sum());i=int(torch.isinf(value).sum())
                    nan_count+=n;inf_count+=i
                    record.update(classification='NONFINITE_GRADIENT',NaN_elements=n,Inf_elements=i)
        if len(records)<max_parameter_records:records.append(record)
    elapsed=time.monotonic()-start
    if elapsed>wall_budget_s:complete=False
    return{'status':'COMPLETE_DECLARED_GRADIENT_OBSERVATION'if complete else'PARTIAL_DIAGNOSTIC_WALL_BUDGET',
      'phase':phase,'actual_scaler_scale':float(scale),'actual_scaler_public_state':dict(state),
      'requires_grad_parameters_observed':total,'missing_gradient_parameters':missing,
      'nonfinite_gradient_parameters':nonfinite,'finite_present_gradient_parameters':finite_count,
      'NaN_gradient_elements':nan_count,'Inf_gradient_elements':inf_count,
      'unreviewed_gradient_layout_or_dtype':unsupported,'gradient_elements_observed':elements,
      'gradient_dtype_parameter_counts':dtype_counts,'parameter_records':records,
      'parameter_records_omitted':max(0,total-len(records)),'wall_s':elapsed,'wall_budget_s':wall_budget_s,
      'all_observed_required_gradients_finite_and_present':complete and total>0 and missing==nonfinite==unsupported==0,
      'model_scaler_optimizer_changed_by_observer':False,'parameter_updates_proven':False,
      'native_training_qualified':False,'scope':'read-only scalar evidence; no overflow-cause or recovery policy inferred'}
