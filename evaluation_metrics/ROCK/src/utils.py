"""
utils.py
"""
import ast, re
import numpy as np
import os, sys, multiprocessing, logging

from collections import defaultdict
import nltk
import spacy
import lemminflect
import torch
from transformers import GPT2Tokenizer, XLNetTokenizer, TransfoXLTokenizer, OpenAIGPTTokenizer
from transformers import GPT2LMHeadModel, XLNetLMHeadModel, TransfoXLLMHeadModel, OpenAIGPTLMHeadModel

"""
generate temporal probabilities
"""
import string
def crop_sent(s, spacy_model, sent_idx=0, offset=3):
    try:
        ss = list(spacy_model(s[s.find("Before that,")+13:].strip().replace('\n', ' ')).sents)[sent_idx].text
    except:
        print(s)
        return s.strip().replace('\n', ' ').strip()

    ss = ss.strip().replace('\n', ' ').strip()    # just strip the white space, replace the newline, and don't remove important punctuations.
    
    try:
        return ss[0].upper() + ss[1:]
    except:
        print(s)
        return ss
    

def conan_get_probs(s, model, spacy_model, top_k=5, tmp_bsz=8):
    
    D, X, Dint, D_d, X_d, Dint_d, D_a, X_a, Dint_a, Y = s[[
        'cause', 'covariates', 'interventions',
        'd_cause', 'd_covariates', 'd_interventions',
        'a_cause', 'a_covariates', 'a_interventions',
        'effect'
    ]]

    # preprocess
    X = [crop_sent(x, spacy_model=spacy_model) for x in X]
    X_d = [crop_sent(x, spacy_model=spacy_model) for x in X_d]
    X_a = [crop_sent(x, spacy_model=spacy_model) for x in X_a]
    concat_X = X+X_d+X_a
    concat_D = [D]+Dint+[D_d]+Dint_d+[D_a]+Dint_a

    # concatenate all pairs
    instances = \
        list(map(lambda f: (f, ""), concat_X))+\
        [(i, j) for i in X for j in [D]+Dint]+\
        [(i, j) for i in X_d for j in [D_d]+Dint_d]+\
        [(i, j) for i in X_a for j in [D_a]+Dint_a]+\
        list(map(lambda f: (f, Y), concat_D+concat_X))

    # batch prediction
    prob_pair = model.get_temp_batch(instances, batch_size=tmp_bsz)

    # locate the index in the concatenated instances
    length_list = np.array([
        len(X), len(X_d), len(X_a),
        len(X)*len([D]+Dint), len(X_d)*len([D_d]+Dint_d), len(X_a)*len([D_a]+Dint_a),
        len([D]+Dint), len([D_d]+Dint_d), len([D_a]+Dint_a),
        len(X), len(X_d), len(X_a),
    ])
    end = np.cumsum(length_list)
    start = end-length_list

    # p(x, d), just covariates, [(x "", "" x)]*cov_size, 30*2
    # p(d, y), covariates and cause+interventions, [[(x1 d1, d1 x1, x1 "", "" x1), (x1 d2, d2 x1, ...), ... to (...d11...)], [(x2 d1, ...), ...], ... to x30], 30*11*4
    # p(x, y), covariates and outcome 30*2
    baseln_probs, baseln_probs_d, baseln_probs_a,\
    tmp, tmp_d, tmp_a,\
    tmp_y_probs, tmp_y_probs_d, tmp_y_probs_a,\
    tmp_xy_probs, tmp_xy_probs_d, tmp_xy_probs_a = \
        [prob_pair[np.arange(st, ed)] for st, ed in zip(start, end)]

    tmp_probs = [[np.hstack((tmp[i*len([D]+Dint)+j], baseln_probs[i])) for j in range(len([D]+Dint))] for i in range(len(X))]
    tmp_probs_d = [[np.hstack((tmp_d[i*len([D_d]+Dint_d)+j], baseln_probs_d[i])) for j in range(len([D_d]+Dint_d))] for i in range(len(X_d))]
    tmp_probs_a = [[np.hstack((tmp_a[i*len([D_a]+Dint_a)+j], baseln_probs_a[i])) for j in range(len([D_a]+Dint_a))] for i in range(len(X_a))]
    
    s['p_xd'] = tmp_probs
    s['p_dy'] = tmp_y_probs
    s['p_xy'] = tmp_xy_probs
    s['p_xd_d'] = tmp_probs_d
    s['p_dy_d'] = tmp_y_probs_d
    s['p_xy_d'] = tmp_xy_probs_d
    s['p_xd_a'] = tmp_probs_a
    s['p_dy_a'] = tmp_y_probs_a
    s['p_xy_a'] = tmp_xy_probs_a
    return s
