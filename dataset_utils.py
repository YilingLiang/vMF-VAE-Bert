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
""" Classification fine-tuning: utilities to work with various datasets """

from __future__ import absolute_import, division, print_function

import csv
from tqdm import tqdm
import logging
import os
import sys
from io import open
import torch
import enum
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class SNLIProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._generate_train_example(data_dir, 'train')

    def get_dev_examples(self, data_dir):
        return self._generate_train_example(data_dir, 'test')

    def _generate_train_example(self, data_dir, sub_dir):
        labels = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        sentence_ids = {}
        lower = True
        examples = []

        sentence_path = os.path.join(data_dir, 'sentences.dlnlp')
        with open(sentence_path, 'r') as f: # sentences.dlnlp
            for lines in f:
                toks=lines.strip().split('\t') # 0  She was wearing a dress.
                sentence_ids[toks[0]]=toks[1].strip() # sentence_ids 是个字典, {'0', 'A person on a horse jumps over a broken down airplane .'}
                # line = sentence_ids[toks[0]]
        data_path = os.path.join(data_dir, sub_dir + '.txt')
        with open(data_path, 'r') as f:
            linecount = 0
            for line in f:
                linecount += 1
                tokens = line.strip().split('\t')
                label = labels[tokens[0]]
                premise = sentence_ids[tokens[1]]
                hypothesis = sentence_ids[tokens[2]]
                if lower:
                    hypothesis = hypothesis.strip().lower()
                    premise = premise.strip().lower()

                examples.append(InputExample(text_a=premise, text_b=hypothesis, label=label))
        return examples

    def get_labels(self):
        return [0, 1, 2]

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 1000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[int(example.label)] # 加了 int
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features

def load_and_cache_examples(task, tokenizer, evaluate=False):
    data_dir = './data/classifier'
    max_seq_length = 64
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file

    logger.info("Creating features from dataset file at %s", data_dir)
    label_list = processor.get_labels()
    examples = processor.get_dev_examples(data_dir) if evaluate else processor.get_train_examples(data_dir)
    features = convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_mode,
                                            cls_token=tokenizer.cls_token,
                                            sep_token=tokenizer.sep_token,
                                            )


    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset

class QQPProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._generate_train_example(data_dir, 'train')

    def get_dev_examples(self, data_dir):
        return self._generate_train_example(data_dir, 'test')

    def _generate_train_example(self, data_dir, sub_dir):
        labels = {'non-duplicate': 0, 'duplicate': 1}
        lower = True
        examples = []

        data_path = os.path.join(data_dir, sub_dir + '.tsv')
        with open(data_path, 'r') as f:
            linecount = 0
            for line in f:
                if linecount > 0:
                    linecount += 1
                    tokens = line.strip().split('\t')
                    try:
                        label = int(tokens[5])
                        premise = tokens[3]
                        hypothesis = tokens[4]
                        # print(premise_ori)
                    except:
                        continue
                else:
                    linecount += 1
                    continue
                if lower:
                    hypothesis = hypothesis.strip().lower()
                    premise = premise.strip().lower()

                examples.append(InputExample(text_a=premise, text_b=hypothesis, label=label))
        return examples

    def get_labels(self):
        return [0, 1]

def load_and_cache_examples_qqp(task, tokenizer, evaluate=False):
    data_dir = './data/QQP'
    max_seq_length = 64
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file

    logger.info("Creating features from dataset file at %s", data_dir)
    label_list = processor.get_labels()
    examples = processor.get_dev_examples(data_dir) if evaluate else processor.get_train_examples(data_dir)
    features = convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_mode,
                                            cls_token=tokenizer.cls_token,
                                            sep_token=tokenizer.sep_token,
                                            )


    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    # f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": acc, # f1,
        "acc_and_f1": acc, # (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "snli":
        return acc_and_f1(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)

processors = {
    "snli": SNLIProcessor,
    "qqp": QQPProcessor,
}

output_modes = {
    "snli": "classification",
    "qqp": "classification",
}

task_labels = {
    "neg": "0",
    "pos": "1",
}
