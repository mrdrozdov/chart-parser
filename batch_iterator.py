from collections import OrderedDict

import numpy as np

import torch
from torch.utils.data import Sampler


class SimpleSampler(Sampler):

    def __init__(self, data_source, batch_size, include_partial=False, rng=None):
        self.data_source = data_source
        self.active = False
        if rng is None:
            rng = np.random.RandomState(seed=11)
        self.rng = rng
        self.batch_size = batch_size
        self.include_partial = include_partial
        self.epoch = 0

    def reset(self):
        order = list(range(len(self.data_source)))
        self.rng.shuffle(order)
        self.order = order
        self.index = -1

    def get_next_batch(self):
        index = self.index + 1

        batch_size = self.batch_size
        start = index * batch_size
        batch_index = self.order[start:start+batch_size]

        self.index = index

        return batch_index

    def __iter__(self):
        self.reset()

        for _ in range(len(self)):
            yield self.get_next_batch()

        self.epoch += 1

    def __len__(self):
        length = len(self.order) // self.batch_size
        if self.include_partial and length * self.batch_size < len(self.order):
            length += 1
        return length


class FixedLengthSampler(Sampler):

    def __init__(self, data_source, batch_size, include_partial=False, rng=None):
        self.data_source = data_source
        self.active = False
        if rng is None:
            rng = np.random.RandomState(seed=11)
        self.rng = rng
        self.batch_size = batch_size
        self.include_partial = include_partial
        self.epoch = 0

    def reset(self):
        length_map = OrderedDict()
        for i in range(len(self.data_source)):
            x = self.data_source.dataset[i]
            length_map.setdefault(len(x), []).append(i)

        # Shuffle the order.
        for length in length_map.keys():
            self.rng.shuffle(length_map[length])

        # Initialize state.
        state = {}
        for length, arr in length_map.items():
            batch_size = self.batch_size
            nbatches = len(arr) // batch_size
            surplus = nbatches * batch_size < len(arr)
            state[length] = dict(nbatches=nbatches, surplus=surplus, position=-1)

        # Batch order, in terms of length.
        order = []
        for length, v in state.items():
            order += [length] * v['nbatches']

        ## Optionally, add partial batches.
        if self.include_partial:
            for length, v in state.items():
                if v['surplus']:
                    order += [length]

        self.rng.shuffle(order)
        self.length_map = length_map
        self.state = state
        self.order = order
        self.index = -1

    def get_next_batch(self):
        index = self.index + 1
        length = self.order[index]
        batch_size = self.batch_size
        position = self.state[length]['position'] + 1
        start = position * batch_size
        batch_index = self.length_map[length][start:start+batch_size]

        self.state[length]['position'] = position
        self.index = index

        return batch_index

    def __iter__(self):
        self.reset()

        for _ in range(len(self)):
            yield self.get_next_batch()

        self.epoch += 1

    def __len__(self):
        return len(self.order)


class SimpleDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        item = self.dataset[index]
        return index, item

    def __len__(self):
        return len(self.dataset)


class BatchIterator(object):
    def __init__(self, sentences, extra={},
                 batch_size=10,
                 seed=121,
                 include_partial=False,
                 num_workers=2,
                 ):
        self.sentences = sentences
        self.extra = extra
        self.loader = None

        self.batch_size = batch_size
        self.seed = seed
        self.include_partial = include_partial
        self.num_workers = num_workers

    def get_iterator(self):
        batch_size = self.batch_size
        seed = self.seed
        include_partial = self.include_partial
        num_workers = self.num_workers

        def collate_fn(batch):
            """
            Executes across multiple process.
            """
            index, sentences = zip(*batch)

            batch_map = {}
            batch_map['index'] = index
            batch_map['sentences'] = sentences

            for k, v in self.extra.items():
                batch_map[k] = [v[idx] for idx in index]

            return batch_map

        if self.loader is None:
            rng = np.random.RandomState(seed=self.seed)
            dataset = SimpleDataset(self.sentences)
            sampler = FixedLengthSampler(dataset, batch_size=batch_size, rng=rng, include_partial=include_partial)
            loader = torch.utils.data.DataLoader(dataset, shuffle=(sampler is None), num_workers=self.num_workers, batch_sampler=sampler, collate_fn=collate_fn)
            self.loader = loader

        def myiterator():
            """
            Executes on main process.
            """
            for i, batch_map in enumerate(self.loader):
                yield batch_map

        return myiterator()
