import numpy as np
import torch
from copy import deepcopy
from torch.autograd import Variable
#! -*- coding: utf-8 -*-
# UniVAE参考实现
# 链接：https://kexue.fm/archives/8475

import json
import numpy as np
import tensorflow as tf
from bert4keras.backend import keras, K
from bert4keras.layers import Loss, integerize_shape
from bert4keras.models import build_transformer_model, RoFormer, BERT
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import DataGenerator, sequence_padding
from bert4keras.snippets import AutoRegressiveDecoder, text_segmentate
from keras.layers import Input, Dense, Lambda, Concatenate, Layer
from keras.models import Model
from UniVAE import UniVAE



def get_min(indices_adv1, indices_adv2, d):
    d1 = deepcopy(d)
    d2 = deepcopy(d)
    idx_adv1 = indices_adv1[np.argmin(d1[indices_adv1])]
    idx_adv2 = indices_adv2[np.argmin(d2[indices_adv2])]
    cnt = 0
    orig_idx_adv1 = idx_adv1
    orig_idx_adv2 = idx_adv2
    while idx_adv1 == idx_adv2 and cnt < 20:
        d1[idx_adv1] = 9999
        d2[idx_adv2] = 9999
        idx_adv1 = indices_adv1[np.argmin(d1[indices_adv1])]
        idx_adv2 = indices_adv2[np.argmin(d2[indices_adv2])]
        cnt+=1
    if cnt == 20:
        return orig_idx_adv1, orig_idx_adv2
    else:
        return idx_adv1, idx_adv2

def get_min_single(indices_adv1, d):
    d1 = deepcopy(d)
    idx_adv1 = indices_adv1[np.argmin(d1[indices_adv1])]
    cnt = 0
    orig_idx_adv1 = idx_adv1
    while cnt < 20:
        d1[idx_adv1] = 9999
        idx_adv1 = indices_adv1[np.argmin(d1[indices_adv1])]
        cnt+=1
    if cnt == 20:
        return orig_idx_adv1
    else:
        return idx_adv1

def convert_sentence2ids(premise_words_ori, hypothesis_words_ori):
    
    vocab = json.load(open('./vocab.json', 'r'))
    maxlen = 32
    premise_words = ['<sos>'] + premise_words_ori
    premise_words += ['<eos>']
    hypothesis_words = ['<sos>'] + hypothesis_words_ori
    unk_idx = vocab['<oov>']
    hypothesis_indices = [vocab[w] if w in vocab else unk_idx for w in hypothesis_words]
    premise_indices = [vocab[w] if w in vocab else unk_idx for w in premise_words]
    if len(premise_indices) < maxlen:
        premise_indices += [0]*(maxlen- len(premise_indices))
    if len(hypothesis_indices) < maxlen:
        hypothesis_indices += [0]*(maxlen - len(hypothesis_indices))
    premise_indices = premise_indices[:maxlen]
    hypothesis_indices = hypothesis_indices[:maxlen] 
    return premise_indices, hypothesis_indices
    
    
def search_qqp_snli(pred_fn_single, s, y, nsamples=100, l=0., h=1.0):
    
    
    premise_word_ori, hypothesis_word_ori = s
    
    x, s = tokenizer.encode(hypothesis_word_ori, maxlen=maxlen)
    X = sequence_padding([x])
    S = sequence_padding([s])
    Z = encoder.predict([X, S])
    z = Z[0]
    x_adv = []
    for k in range(nsamples):

        delta_z = np.random.randn(64)
        d = np.random.rand(64) * (h - l) + l  # length range [l, h)
        norm_p = np.linalg.norm(delta_z, ord=2)

        d_norm = np.divide(d, norm_p)  # rescale/normalize factor
        delta_z = np.multiply(delta_z, d_norm) 
        delta_z += z

        hypo_pre_ori = vec2sent.generate(delta_z)
        p, h = convert_sentence2ids(premise_words_ori, hypo_pre_ori)
        y_pred = pred_fn_one((p.unsqueeze(0), h.unsqueeze(0)))
        if y_pred != y:
            x_adv.append(premise_word_ori + '\t' + hypo_pre_ori + '\n')
    return x_adv
        


def random_smoothing(generator, z, nsamples=100, l=0., h=1.0, p=2):
    # x 是 (premise，hypothesis) 元组， y 是真标签， z 是 假设的隐编码， 我们扰动的是假设(hypothesis)的隐编码找对抗样本

    delta_z = np.random.randn(nsamples, z.shape[1])  # http://mathworld.wolfram.com/HyperspherePointPicking.html
    d = np.random.rand(nsamples) * (h - l) + l  # length range [l, h)
    norm_p = np.linalg.norm(delta_z, ord=p, axis=1)
    d_norm = np.divide(d, norm_p).reshape(-1, 1)  # rescale/normalize factor
    delta_z = np.multiply(delta_z, d_norm)  # delta_z * d /|delta_z|
    delta_z = torch.FloatTensor(delta_z) # 加的
    delta_z += z  # z tilde
    x_adv_v = generator(delta_z)  # x tilde

    return x_adv_v