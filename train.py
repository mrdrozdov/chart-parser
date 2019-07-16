import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from penn_treebank_reader import *
from dataset import DatasetReader
from dataset import make_batch_iterator


def get_offset_cache(length):
    offset_cache = {}
    ncells = int(length * (1 + length) / 2)
    for lvl in range(length):
        level_length = length - lvl
        ncells_less = int(level_length * (1 + level_length) / 2)
        offset_cache[lvl] = ncells - ncells_less
    return offset_cache


class ChartUtils(object):
    def __init__(self):
        super(ChartUtils, self).__init__()
        self.offset_cache = {}

    def to_idx(self, length, level, pos):
        return self.to_offset(length, level) + pos

    def to_offset(self, length, level):
        if length not in self.offset_cache:
            self.offset_cache[length] = get_offset_cache(length)
        return self.offset_cache[length][level]
chart_utils = ChartUtils()


class ModelContainer(nn.Module):
    def __init__(self, embed, model, loss_func):
        super(ModelContainer, self).__init__()
        self.embed = embed
        self.model = model
        self.loss_func = loss_func


class ConstituentLoss(nn.Module):
    def __init__(self, vocab_size):
        super(ConstituentLoss, self).__init__()
        self.hidden_size = 100
        self.predict = nn.Linear(self.hidden_size, vocab_size)
        
    def forward(self, chart, label_batch):
        batch_index = label_batch['batch_index']
        idx_index = label_batch['idx_index']
        label_index = label_batch['label_index']

        logit = self.predict(chart[batch_index, idx_index])
        loss = nn.CrossEntropyLoss()(logit, label_index)

        return loss


class SequenceEncoder(nn.Module):
    def __init__(self, vocab_size, self_attention=False):
        super(SequenceEncoder, self).__init__()
        self.self_attention = self_attention
        self.hidden_size = 100
        self.embed = nn.Embedding(vocab_size, self.hidden_size)

        if self.self_attention:
            self.atten_q = nn.Linear(self.hidden_size, self.hidden_size)

    def run_attention(self, h):
        q, k, v = h, h, h
        scores = torch.matmul(self.atten_q(q), k.transpose(1, 2))
        return torch.sum(scores.unsqueeze(3) * v.unsqueeze(2), 2)

    def forward(self, x):
        h = self.embed(x)
        if self.self_attention:
            h = self.run_attention(h)
        return h


class ChartEncoder(nn.Module):
    def __init__(self):
        super(ChartEncoder, self).__init__()
        self.hidden_size = 100
        self.compose = nn.Sequential(
            nn.Linear(2*self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size))
        self.score = nn.Linear(2*self.hidden_size, 1) # TODO: Use dot product instead.

    def step(self, level):
        N = level # number of constituent pairs.
        L = self.length - level # number of elements being computed.

        l_index, r_index = [], []
        ref_idx, ref_pos = [], []

        for idx in range(N):
            left_level = idx
            right_level = level - idx - 1
            left_offset = chart_utils.to_offset(self.length, left_level)
            right_offset = chart_utils.to_offset(self.length, right_level + 1) - L
            left_index = torch.arange(left_offset, left_offset+L)
            right_index = torch.arange(right_offset, right_offset+L)

            l_index.append(left_index)
            r_index.append(right_index)
            ref_idx.append(torch.LongTensor([idx]*L))
            ref_pos.append(torch.arange(L))

        l_index = torch.cat(l_index)
        r_index = torch.cat(r_index)
        ref_idx = torch.cat(ref_idx)
        ref_pos = torch.cat(ref_pos)

        l = self.chart.index_select(index=l_index, dim=1)
        r = self.chart.index_select(index=r_index, dim=1)
        state = torch.cat([l, r], 2)
        h_raw = self.compose(state)
        s_raw = self.score(state) # TODO: Should incorporate score from children.

        s = torch.softmax(s_raw.view(self.batch_size, L, N, 1), dim=2)
        hbar = torch.sum(s * h_raw.view(self.batch_size, L, N, self.hidden_size), 2)
        # sbar = torch.sum(s * s_raw.view(self.batch_size, L, N, 1), 2)

        offset = chart_utils.to_offset(self.length, level)

        self.chart[:, offset:offset+L] = hbar

    def build_chart(self, x):
        chart_size = self.length * (self.length + 1) // 2
        chart = torch.FloatTensor(self.batch_size, chart_size, self.hidden_size).fill_(0)
        chart[:, :self.length] = x
        self.chart = chart

        for level in range(1, self.length):
            self.step(level)

    def init_batch(self, x):
        self.batch_size = x.shape[0]
        self.length = x.shape[1]
        
    def forward(self, x):
        self.init_batch(x)
        self.build_chart(x)
        return None


class BatchManager(object):
    def prepare_batch(self, batch_map):
        return torch.LongTensor(batch_map['sentences'])

    def prepare_labels(self, batch_map):
        batch_index = []
        idx_index = []
        label_index = []

        length = len(batch_map['sentences'][0])

        for s in batch_map['sentences']:
            assert len(s) == length, 'Does not support variable length batches.'

        for i, spans in enumerate(batch_map['labels']):
            for pos, size, label in spans:
                level = size - 1
                batch_index.append(i)
                idx_index.append(chart_utils.to_idx(length, level, pos))
                label_index.append(label)

        batch_index = torch.LongTensor(batch_index)
        idx_index = torch.LongTensor(idx_index)
        label_index = torch.LongTensor(label_index)

        return {
            'batch_index': batch_index,
            'idx_index': idx_index,
            'label_index': label_index,
        }


def main(options):
    tr_reader = JSONLReader(options.tr_file)
    tr_dataset = DatasetReader(tr_reader, config={'max_len': options.tr_max_len}).build()
    batch_iterator = make_batch_iterator(None, tr_dataset)
    embed = SequenceEncoder(self_attention=options.self_attention, vocab_size=len(tr_dataset['metadata']['word2idx']))
    model = ChartEncoder()
    loss_func = ConstituentLoss(vocab_size=len(tr_dataset['metadata']['label2idx']))
    container = ModelContainer(embed, model, loss_func)
    params = container.parameters()
    optimizer = optim.Adam(params, lr=0.002, betas=(0.9, 0.999), eps=1e-8)

    print('# of sentences = {}'.format(len(tr_dataset['sentences'])))
    print('vocab size = {}'.format(len(tr_dataset['metadata']['word2idx'])))
    print('# of classes = {}'.format(len(tr_dataset['metadata']['label2idx'])))
    print(tr_dataset['metadata']['label2idx'])

    for epoch in range(options.max_epochs):
        for batch_map in batch_iterator.get_iterator():
            seq = BatchManager().prepare_batch(batch_map)
            seqh = embed(seq)
            _ = model(seqh)

            label_batch = BatchManager().prepare_labels(batch_map)
            loss = loss_func(model.chart, label_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()

            print(loss.item())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--tr_file', default=os.path.expanduser('~/data/ptb/valid.jsonl'), type=str)
    parser.add_argument('--tr_max_len', default=10, type=int)
    parser.add_argument('--self_attention', action='store_true')
    parser.add_argument('--max_epochs', default=1000, type=int)
    options = parser.parse_args()

    main(options)
