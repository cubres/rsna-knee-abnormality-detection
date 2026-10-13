#!/usr/bin/env python3
"""Render selected actual synthetic goldens; no clinical data or score claims. MIT."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('corpus');p.add_argument('--out',required=True);a=p.parse_args()
root=Path(a.corpus);out=Path(a.out)
if out.exists(): raise FileExistsError('Choose a fresh figure output')
cases=[('ramp_8x5_u16_m2','Row / column order'),('ramp_8x5_u16_m1','MONOCHROME1 inversion'),('stored12_i16_m2','12-bit signed decoding'),('rescale_i16_2','Negative slope + window'),('window_width5_m2','LINEAR window boundary'),('signed_endpoints','Signed 16-bit endpoints')]
fig,axes=plt.subplots(2,3,figsize=(13.6,7.8),facecolor='#101a32')
fig.suptitle('Small DICOM fixtures. Observable preprocessing bugs.',color='white',fontsize=22,fontweight='bold',y=.97)
for ax,(name,label) in zip(axes.flat,cases):
    image=np.load(root/'golden'/f'{name}.npy',allow_pickle=False)
    ax.imshow(image,cmap='gray',vmin=0,vmax=1,interpolation='nearest',aspect='auto')
    ax.set_title(label,color='white',fontsize=13,pad=10)
    ax.set_xticks([]);ax.set_yticks([])
    ax.set_xlabel(name,color='#b8cee7',fontsize=10)
    for spine in ax.spines.values():spine.set_edgecolor('#527493')
fig.text(.5,.035,'48 invented encoding fixtures | 43 analytical goldens | 5 expected refusals | No patient images',ha='center',color='#b8cee7',fontsize=13)
fig.subplots_adjust(top=.86,bottom=.12,left=.045,right=.955,hspace=.42,wspace=.16)
with out.open('xb') as f:fig.savefig(f,format='png',dpi=150,facecolor=fig.get_facecolor())
plt.close(fig)
