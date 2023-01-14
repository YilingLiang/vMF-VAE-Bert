# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on various datasets (Bert, XLM, XLNet)."""



from __future__ import absolute_import, division, print_function


import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
# from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, BertConfig # from pretrained
from utils import to_gpu, Corpus, batchify, SNLIDataset, collate_snli
from dataset_utils import convert_examples_to_features, load_and_cache_examples, SNLIProcessor
from dataset_utils import (compute_metrics, convert_examples_to_features,
                        output_modes, processors)
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import sys
import json
sys.path.append("..") # 导入上级目录模块
# print("系统当前路径" , sys.path)
task_name = 'snli'
# model_name_or_path = './bert-base-uncased'
model_name_or_path = './bert_snli_uncased/checkpoint-25752'
epochs = 3
num_labels = 3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
}
# BertConfig 在transformer 中的才有 from_pretrained
config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']

config = config_class.from_json_file(model_name_or_path+ '/config.json')
print(config)
tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=True)

model = model_class.from_pretrained(model_name_or_path, num_labels=3)
model.to(device)
train_iterator = trange(int(epochs), desc="Epoch")

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)

def evaluate_model():
    eval_dataset = load_and_cache_examples(task_name, tokenizer, evaluate=True)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=64)
    eval_loss = 0.0
    eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      # XLM don't use segment_ids
                      'labels': batch[3]}
            outputs = model(**inputs)
            eval_steps += 1
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()

            if preds is None:
                preds = logits.detach().cpu().numpy()
                # print(preds.shape)
                out_label_ids = inputs['labels'].detach().cpu().numpy()
                # print(out_label_ids.shape)
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
    print(preds.shape)
    preds = np.argmax(preds, axis=1)
    result = compute_metrics(task_name, preds, out_label_ids)
    loss = eval_loss / eval_steps
    print('loss:',loss, '  acc:', result)
    return result

best_acc = 0.85



train_mode = True
if train_mode == True:

    global_steps = 0
    tr_loss = 0
    for _ in train_iterator:
        # generate perturbed training set
        train_dataset = load_and_cache_examples(task_name, tokenizer, evaluate=False)

        train_sampler = RandomSampler(train_dataset)
        # train_dataloader has been tokenized here
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=64)

        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            global_steps += 1
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      # XLM don't use segment_ids
                      'labels': batch[3]}
            ouputs = model(**inputs)
            loss = ouputs[0]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            tr_loss += loss.item()
            if global_steps % 2000 == 0:
                print(tr_loss / global_steps)
            optimizer.step()
            model.zero_grad()
            # ======测试代码之后删====
            # evaluate_model()
            # tokenizer.save_pretrained('./')
            # torch.save(model.state_dict(), 'bertsnli.pt')
            # model.save_pretrained('./') #两个的保存效果一样，model.save_pretrained()会自动命名为 pytorch_model.bin, 另外会save一个config.json
            # ======测试代码之后删====
        
        # save model
        output_dir = './bert_snli_uncased'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        result = evaluate_model()
        # 保存最好的模型
        if result['acc'] > best_acc:
            best_acc = result['acc']
            save_dir = os.path.join(output_dir, 'checkpoint-{}-{}'.format(global_steps, best_acc))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print("Saving model checkpoint to %s", save_dir)
    # 保存最后一个模型        
    last_dir = os.path.join(output_dir, 'last-{}'.format(result['acc']))
    if not os.path.exists(last_dir):
        os.makedirs(last_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(last_dir)
    tokenizer.save_pretrained(last_dir)
else:
    evaluate_model()