"""
metrics.py
"""
import numpy as np
import pandas as pd

def return_match_idx(mat, ord=None, eps=0.1, return_arr=False):
    d0 = mat[:,0].T
    ds = mat[:,1:].T
    Z = mat.shape[0]
    arr = np.linalg.norm(ds - d0, ord=ord, axis=1)/Z#(mat.shape[1]-1)#mat.shape[0]
    mat_idx = np.where( (arr < eps) & (arr > 1e-8))[0]
    if return_arr:
        return mat_idx, arr
    return mat_idx

def delta_bar(p_xd, p_dy, p_xy, eps=0.01, ord=1, normalization=True,
              direct_match=False, use_cooccur=False, temp_filter=False, 
              res_norm=False, pxa_all_norm=False,
              **kwargs):
    
    def adj(tprobs, eps, ord, temp_filter=False):
        if temp_filter:
            tprobs = tprobs[tprobs[:,0,0]>tprobs[:,0,1]]
        Zn = tprobs[:,:,1]+ tprobs[:,:,0]+tprobs[:,:,2]+ tprobs[:,:,3]
        Zn[np.where(Zn==0)] = 1
        dat = tprobs[:,:,0]
        if normalization:
            dat /= Zn

        
        p_x = dat[:,0]/np.sum(dat[:,0])
        if pxa_all_norm:
            p_xa = dat/np.sum(dat, axis=0)
            p_xa[:,0] = p_x
        else:
            p_xa = dat / np.sum(dat[:,0])
            
        # in case some covariate has 0 probability of occuring,
        # set it to uniform
        p_x[p_x==0]=1/(dat.shape[0] if dat.shape[0] != 0 else 1)

        
        p_a_x = (p_xa.T/(p_x.T)).T

        matched_idx = return_match_idx(p_a_x, ord=ord, eps=eps)

        return matched_idx
    if direct_match:
        matched_idx = return_match_idx(p_xd[:,:,0], ord=ord, eps=eps)        
    else:
        matched_idx = adj(p_xd, eps=eps, ord=ord, temp_filter=False, **kwargs)

    if use_cooccur:
        sc = np.mean(p_dy[0,:2]) - (np.nanmean(p_dy[1:,:2][matched_idx]) if len(matched_idx) > 0 else 0)
    elif res_norm:
        def div_0arr(s):
            s[s==0]=1
            return s
        div_0 = lambda s : s if s != 0 else 1
        p_dy_proc = p_dy[1:,0] / div_0arr(p_dy[1:,0]+p_dy[1:,1])
        p_dy_0 = p_dy[0,0] / div_0(p_dy[0,0]+p_dy[0,1])
        sc = p_dy_0 - (np.nanmean(p_dy_proc[matched_idx]) if len(matched_idx) > 0 else 0)
    else:
        sc = p_dy[0,0] - (np.nanmean(p_dy[1:,0][matched_idx]) if len(matched_idx) > 0 else 0)

    return sc
